"""Model performance dashboard displaying frozen held-out test benchmark metrics."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def render_performance_dashboard() -> None:
    """Render historical held-out test performance metrics and confusion analysis."""
    st.markdown("## 📊 Model Performance & Historical Benchmark Metrics")
    st.caption(
        "Official held-out test evaluation results on the sealed WM-811K benchmark partition "
        "(N = 24,705 wafers, 1,537 manufacturing lots, zero lot leakage). "
        "These are historical sealed metrics and are not re-evaluated dynamically."
    )

    # Global KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Test Accuracy",
            "97.62%",
            help="Overall classification accuracy across all 24,705 test wafers.",
        )
    with col2:
        st.metric(
            "Macro F1-Score", "0.8746", help="Unweighted harmonic mean of F1 across all 9 classes."
        )
    with col3:
        st.metric(
            "Weighted F1-Score", "0.9761", help="Class-weighted F1 reflecting dataset frequency."
        )
    with col4:
        st.metric(
            "False Alarm Rate",
            "0.95%",
            help="Normal clean wafers falsely flagged as defect (200 / 21,061).",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Macro Precision", "89.09%", help="Average precision across all 9 classes.")
    with col6:
        st.metric("Macro Recall", "86.22%", help="Average recall across all 9 classes.")
    with col7:
        st.metric(
            "GPU Throughput", "1,848.3 w/s", help="Steady-state inference speed (163.4 µs/wafer)."
        )
    with col8:
        st.metric(
            "Calibration (ECE)",
            "1.64%",
            help="Expected Calibration Error across 15 probability bins.",
        )

    st.markdown("---")

    # Per-Class Performance Table
    st.markdown("### 📋 Complete Per-Class Held-Out Test Breakdown")

    metrics_json_path = Path("data/interim/figures/final_test_metrics.json")
    if metrics_json_path.exists():
        with open(metrics_json_path) as f:
            metrics_data = json.load(f)
        per_class = metrics_data.get("per_class_metrics", {})

        table_rows = []
        for cname, m in per_class.items():
            table_rows.append(
                {
                    "Defect Class": cname,
                    "Support": m["support"],
                    "Precision": f"{m['precision'] * 100:.2f}%",
                    "Recall": f"{m['recall'] * 100:.2f}%",
                    "F1-Score": f"{m['f1']:.4f}",
                    "Clean Escapes": f"{m['escapes_to_clean']} ({m['clean_escape_rate'] * 100:.2f}%)"
                    if m["class_index"] != 0
                    else "N/A (clean)",
                    "Other Defect Misclassifications": m.get("escapes_to_other_defects", 0),
                }
            )

        df_metrics = pd.DataFrame(table_rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("Detailed metrics JSON not found locally. Standard benchmark summary displayed.")

    st.markdown("---")

    # Confusion Matrix Image
    st.markdown("### 🔲 Held-Out Test Confusion Matrix Heatmap")
    cm_img_path = Path("data/interim/figures/final_test_confusion_matrix.png")
    if cm_img_path.exists():
        st.image(
            str(cm_img_path),
            caption="Held-Out Test Partition Confusion Matrix (24,705 Wafers)",
            use_container_width=True,
        )
    else:
        st.warning(f"Confusion matrix plot not found at {cm_img_path}.")
