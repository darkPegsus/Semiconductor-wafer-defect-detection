"""Model performance dashboard displaying frozen held-out test benchmark metrics."""

import json
from pathlib import Path

import streamlit as st


def render_performance_dashboard() -> None:
    """Render historical held-out test performance metrics and confusion analysis."""
    st.markdown("### 📊 Model Performance | Historical Benchmark Evaluation")

    st.markdown(
        """
        <div style="background-color: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 700; color: #0369a1; font-size: 0.95rem;">
                📌 Official Sealed Test Benchmark Notice
            </div>
            <div style="color: #334155; font-size: 0.88rem; margin-top: 4px;">
                The metrics below represent the <strong>final held-out test partition evaluation</strong> on the WM-811K benchmark
                (<strong>N = 24,705 wafers</strong> across <strong>1,537 lots</strong> with verified zero lot leakage).
                These are fixed historical verification metrics establishing model capability, <strong>not dynamic measurements of the current uploaded wafer</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Global KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Test Accuracy",
            "97.62%",
            help="Overall classification accuracy across all 24,705 test wafers.",
        )
    with col2:
        st.metric(
            "Macro F1-Score",
            "0.8746",
            help="Unweighted harmonic mean of F1 across all 9 classes.",
        )
    with col3:
        st.metric(
            "Macro Precision",
            "89.09%",
            help="Average precision across all 9 classes.",
        )
    with col4:
        st.metric(
            "Macro Recall",
            "86.22%",
            help="Average recall across all 9 classes.",
        )
    with col5:
        st.metric(
            "Weighted F1-Score",
            "0.9761",
            help="Class-weighted F1 reflecting dataset frequency distribution.",
        )

    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric(
            "Calibration Error (ECE)",
            "1.64%",
            help="Expected Calibration Error across 15 probability bins.",
        )
    with col7:
        st.metric(
            "Clean False Alarm Rate",
            "0.95%",
            help="Clean wafers falsely classified as defects (200 / 21,061).",
        )
    with col8:
        st.metric(
            "GPU Steady-State Speed",
            "1,848.3 w/s",
            help="163.4 µs/wafer throughput in batched GPU inference.",
        )

    st.markdown("---")

    # Per-Class Performance Table
    st.markdown("#### 📋 Complete Per-Class Held-Out Test Breakdown")

    metrics_json_path = Path("data/interim/figures/final_test_metrics.json")
    if metrics_json_path.exists():
        with open(metrics_json_path) as f:
            metrics_data = json.load(f)
        per_class = metrics_data.get("per_class_metrics", {})

        table_rows = []
        for cname, m in per_class.items():
            escape_str = (
                f"{m['escapes_to_clean']} ({m['clean_escape_rate'] * 100:.2f}%)"
                if m["class_index"] != 0
                else "N/A (Clean)"
            )
            escape_color = "#dc2626" if m.get("escapes_to_clean", 0) > 20 else "#475569"
            table_rows.append(
                f"<tr>"
                f"<td style='padding: 10px 14px; font-weight: 700; color: #0f172a; border-bottom: 1px solid #e2e8f0;'>{cname}</td>"
                f"<td style='padding: 10px 14px; color: #334155; border-bottom: 1px solid #e2e8f0;'>{m['support']:,}</td>"
                f"<td style='padding: 10px 14px; font-weight: 600; color: #0284c7; border-bottom: 1px solid #e2e8f0;'>{m['precision'] * 100:.2f}%</td>"
                f"<td style='padding: 10px 14px; font-weight: 600; color: #0284c7; border-bottom: 1px solid #e2e8f0;'>{m['recall'] * 100:.2f}%</td>"
                f"<td style='padding: 10px 14px; font-weight: 700; color: #0f172a; border-bottom: 1px solid #e2e8f0;'>{m['f1']:.4f}</td>"
                f"<td style='padding: 10px 14px; color: {escape_color}; border-bottom: 1px solid #e2e8f0;'>{escape_str}</td>"
                f"<td style='padding: 10px 14px; color: #64748b; border-bottom: 1px solid #e2e8f0;'>{m.get('escapes_to_other_defects', 0)}</td>"
                f"</tr>"
            )

        rows_html = "".join(table_rows)
        table_html = (
            "<div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>"
            "<table style='width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem;'>"
            "<thead>"
            "<tr style='background-color: #f8fafc; border-bottom: 2px solid #e2e8f0; color: #475569; font-weight: 700; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.04em;'>"
            "<th style='padding: 12px 14px;'>Defect Class</th>"
            "<th style='padding: 12px 14px;'>Test Support</th>"
            "<th style='padding: 12px 14px;'>Precision</th>"
            "<th style='padding: 12px 14px;'>Recall</th>"
            "<th style='padding: 12px 14px;'>F1-Score</th>"
            "<th style='padding: 12px 14px;'>Escapes to Clean</th>"
            "<th style='padding: 12px 14px;'>Other Confusion</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "</div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("Detailed metrics JSON not found locally. Standard benchmark summary displayed.")

    st.markdown("---")

    # Confusion Matrix Image
    st.markdown("#### 🔲 Held-Out Test Confusion Matrix Heatmap")
    cm_img_path = Path("data/interim/figures/final_test_confusion_matrix.png")
    if cm_img_path.exists():
        st.image(
            str(cm_img_path),
            caption="Held-Out Test Partition Confusion Matrix (24,705 Wafers, 1,537 Lots)",
            use_container_width=True,
        )
    else:
        st.warning(f"Confusion matrix plot not found at {cm_img_path}.")
