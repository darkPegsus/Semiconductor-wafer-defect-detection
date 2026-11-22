"""Model inference service for CustomWaferCNN wafer defect classification."""

import time
from typing import Any

import torch

from src.data.dataset import IDX_TO_LABEL
from src.models.custom_cnn import CustomWaferCNN

# Physical semiconductor fabrication context for each defect class
DEFECT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "none": {
        "title": "Clean / Normal Production Wafer",
        "mechanism": "Normal fabrication run with acceptable yield. Minor random defects may be present, but no systematic spatial cluster exists.",
        "morphology": "Uniform distribution of functional dies with zero or isolated point failures.",
        "fab_action": "Standard batch release. No fab equipment intervention required.",
    },
    "Center": {
        "title": "Center Radial Defect Cluster",
        "mechanism": "Chemical Mechanical Planarization (CMP) center over-polishing, chemical vapor deposition (CVD) gas nozzle flow variation, or thermal chuck center gradient.",
        "morphology": "Circular or elliptical dense defect core centered at wafer origin.",
        "fab_action": "Inspect CMP head downforce profile and CVD gas delivery center injector.",
    },
    "Donut": {
        "title": "Donut (Toroidal) Annular Defect",
        "mechanism": "Radial slurry pooling during post-CMP spin rinse, or circular thermal gradient during rapid thermal annealing (RTA).",
        "morphology": "Concentric toroidal ring of defective dies surrounding a healthy, functional center.",
        "fab_action": "Check spin-dryer nozzle rotation speed and thermal chamber quartz heating uniformity.",
    },
    "Edge-Loc": {
        "title": "Edge-Localized Defect Cluster",
        "mechanism": "Edge Bead Removal (EBR) solvent splash, robotic edge grip clamp stress, or wafer cassette handling contact.",
        "morphology": "Compact defect cluster located directly along the circumferential wafer boundary.",
        "fab_action": "Audit EBR dispense arm calibration and wafer pre-aligner mechanical grips.",
    },
    "Edge-Ring": {
        "title": "Edge-Ring Perimeter Defect",
        "mechanism": "Plasma sheath edge discontinuity during reactive ion etching (RIE), edge photoresist film thickness drift, or bevel exclusion zone defect.",
        "morphology": "Continuous or semi-continuous ring tracing the 360-degree wafer perimeter.",
        "fab_action": "Calibrate focus ring erosion status and RF plasma chamber edge tuning.",
    },
    "Loc": {
        "title": "Localized Defect Cluster",
        "mechanism": "Foreign particulate splatter, liquid droplet residue, or chuck vacuum pin contamination.",
        "morphology": "Dense, asymmetric spot or island of defective dies away from the center.",
        "fab_action": "Clean electrostatic chuck (ESC) surface and check wet-bench filtration.",
    },
    "Random": {
        "title": "Random Poisson Point Defects",
        "mechanism": "Airborne cleanroom microparticulates, incoming raw silicon ingot crystal dislocations, or gate oxide pinholes.",
        "morphology": "Uniformly dispersed single-die point defects without geometric clustering.",
        "fab_action": "Check cleanroom HEPA filter efficiency and raw substrate incoming quality.",
    },
    "Scratch": {
        "title": "Mechanical Scratch Defect",
        "mechanism": "Robot transfer arm blade contact, vacuum wand slippage, or abrasive foreign grit on CMP polishing pad.",
        "morphology": "Linear, curvilinear, or jagged continuous scratch path cutting across dies.",
        "fab_action": "Immediate inspection of robotic wafer transfer end-effectors and dress/replace CMP pad.",
    },
    "Near-full": {
        "title": "Near-Full Catastrophic Yield Loss",
        "mechanism": "Gross chamber contamination, catastrophic gas delivery starvation, or complete lithographic exposure failure.",
        "morphology": "Nearly the entire wafer surface is defective (>80% die failure).",
        "fab_action": "Immediate tool down-status (E10). Quarantining preceding and following lots.",
    },
}


class WaferInferenceService:
    """Service to execute fast, deterministic, frozen inference."""

    CONFIDENCE_REVIEW_THRESHOLD = 0.85  # Informational review flag for operators

    @classmethod
    def predict(
        cls,
        model: CustomWaferCNN,
        input_tensor: torch.Tensor,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Run model inference on preprocessed wafer tensor.

        Args:
            model: Frozen CustomWaferCNN model instance.
            input_tensor: 4D FloatTensor of shape [1, 1, 64, 64].
            device: Target torch device (CUDA or CPU).

        Returns:
            Dictionary containing prediction, confidence, probability distribution, and metadata.
        """
        if input_tensor.ndim != 4 or input_tensor.shape != (1, 1, 64, 64):
            raise ValueError(f"Expected tensor shape [1, 1, 64, 64], got {input_tensor.shape}")

        target_device = torch.device(device) if isinstance(device, str) else device
        tensor_dev = input_tensor.to(target_device)

        # Synchronize if GPU for precise latency measurement
        if target_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            logits = model(tensor_dev)
            probs_t = torch.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs_t, dim=-1).item())

        if target_device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        probs_np = probs_t.squeeze(0).cpu().numpy()
        pred_label = IDX_TO_LABEL[pred_idx]
        confidence = float(probs_np[pred_idx])

        # Full probability dictionary sorted by class index
        probabilities = {IDX_TO_LABEL[i]: float(probs_np[i]) for i in range(9)}

        # Find second most probable class for ambiguity detection
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        second_class, second_conf = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)

        # Informational review recommendation (confidence < threshold or close second runner-up)
        is_low_confidence = confidence < cls.CONFIDENCE_REVIEW_THRESHOLD
        is_ambiguous = (confidence - second_conf) < 0.25
        review_recommended = is_low_confidence or is_ambiguous

        review_reasons = []
        if is_low_confidence:
            review_reasons.append(
                f"Confidence below nominal review guideline ({confidence * 100:.1f}% < {cls.CONFIDENCE_REVIEW_THRESHOLD * 100:.0f}%)"
            )
        if is_ambiguous and second_class:
            review_reasons.append(
                f"High secondary probability for '{second_class}' ({second_conf * 100:.1f}%)"
            )

        return {
            "predicted_class_idx": pred_idx,
            "predicted_class": pred_label,
            "confidence": confidence,
            "confidence_pct": round(confidence * 100.0, 2),
            "probabilities": probabilities,
            "sorted_probabilities": sorted_probs,
            "latency_ms": round(latency_ms, 3),
            "device": str(target_device),
            "is_clean_wafer": pred_label == "none",
            "review_recommended": review_recommended,
            "review_reasons": review_reasons,
            "description": DEFECT_DESCRIPTIONS.get(pred_label, {}),
            "raw_logits": logits.squeeze(0).cpu().numpy().tolist(),
        }
