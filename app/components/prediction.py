"""Prediction display component with probability charts and neutral inspection notes."""

from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Canonical class list in defined order
CANONICAL_CLASSES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]


def render_prediction_summary(result: dict[str, Any], stats: dict[str, Any] | None = None) -> None:
    """Render prediction metrics and primary classification card.

    Args:
        result: Dictionary output from WaferInferenceService.predict().
        stats: Wafer die statistics dictionary.
    """
    pred_class = result["predicted_class"]
    confidence_pct = result["confidence_pct"]
    is_clean = result["is_clean_wafer"]
    latency_ms = result["latency_ms"]
    device_name = str(result["device"]).upper()
    defect_rate_pct = stats.get("defect_rate_pct", 0.0) if stats else 0.0

    st.markdown("#### 🎯 Classification Results")

    # Primary Status Card
    if is_clean:
        status_bg = "#f0fdf4"
        status_border = "#86efac"
        status_text = "#166534"
        status_title = "NORMAL WAFER (NO SYSTEMATIC DEFECT PATTERN)"
        status_badge = "status-badge-clean"
    else:
        status_bg = "#fef2f2"
        status_border = "#fca5a5"
        status_text = "#991b1b"
        status_title = f"DEFECT PATTERN DETECTED: {pred_class.upper()}"
        status_badge = "status-badge-defect"

    st.markdown(
        f"""
        <div style="background-color: {status_bg}; border: 1px solid {status_border}; border-left: 6px solid {status_text}; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span class="{status_badge}">● {status_title}</span>
                    <h2 style="margin: 6px 0 0 0; color: #0f172a; font-size: 1.85rem; font-weight: 800;">
                        {pred_class}
                    </h2>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Model Confidence (Softmax)</div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: {"#16a34a" if is_clean else "#0284c7"};">
                        {confidence_pct:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Secondary Metric Row
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Inference Latency", f"{latency_ms:.2f} ms", f"Compute: {device_name}")
    with m_col2:
        bad_count = stats.get("defective_dies", 0) if stats else 0
        st.metric("Die Defect Ratio", f"{defect_rate_pct:.2f}%", f"{bad_count:,} defective dies")


def render_probability_chart(probabilities: dict[str, float], predicted_class: str) -> None:
    """Render horizontal bar chart for all nine canonical classes with high contrast.

    Args:
        probabilities: Dictionary mapping class names to probabilities.
        predicted_class: Name of predicted class to highlight.
    """
    classes = [c for c in CANONICAL_CLASSES if c in probabilities]
    probs = [probabilities[c] * 100.0 for c in classes]

    # Clean high-contrast styling: highlight predicted class, mute others
    bar_colors = []
    border_colors = []
    text_colors = []

    for c in classes:
        if c == predicted_class:
            if predicted_class == "none":
                bar_colors.append("#22c55e")  # Green
                border_colors.append("#16a34a")
            else:
                bar_colors.append("#0284c7")  # Clean Sky Blue
                border_colors.append("#0369a1")
            text_colors.append("#ffffff")
        else:
            bar_colors.append("#e2e8f0")  # Soft light slate
            border_colors.append("#cbd5e1")
            text_colors.append("#0f172a")

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker={
                "color": bar_colors,
                "line": {"color": border_colors, "width": 1.5},
            },
            text=[f"<b>{p:.1f}%</b>" if p >= 1.0 else "<1%" for p in probs],
            textposition="auto",
            textfont={
                "color": text_colors,
                "size": 11,
                "family": "-apple-system, BlinkMacSystemFont, Segoe UI",
            },
            hoverinfo="y+x",
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Class Probability Distribution (All 9 Canonical Defect Classes)</b>",
            "font": {"size": 13, "color": "#0f172a"},
        },
        xaxis={
            "title": "<b>Softmax Probability (%)</b>",
            "title_font": {"size": 11, "color": "#475569"},
            "range": [0, 105],
            "gridcolor": "#f1f5f9",
            "tickcolor": "#cbd5e1",
            "tickfont": {"color": "#475569", "size": 11},
            "zerolinecolor": "#cbd5e1",
        },
        yaxis={
            "autorange": "reversed",
            "tickfont": {
                "size": 12,
                "color": "#0f172a",
                "family": "-apple-system, BlinkMacSystemFont, Segoe UI",
            },
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 80, "r": 20, "t": 35, "b": 35},
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_inspection_notes(result: dict[str, Any]) -> None:
    """Render neutral, factual inspection notes and human engineering review guidance.

    Args:
        result: Dictionary output from WaferInferenceService.predict().
    """
    pred_class = result["predicted_class"]
    confidence_pct = result["confidence_pct"]
    sorted_probs = result.get("sorted_probabilities", [])

    second_class, second_conf = ("", 0.0)
    if len(sorted_probs) > 1:
        second_class, second_prob = sorted_probs[1]
        second_conf = second_prob * 100.0

    st.markdown("#### 📋 Inspection Notes & Engineering Guidance")

    # Review advisories based on model state and known benchmark limitations
    review_items = []

    # 1. Low confidence flag (< 85%)
    if confidence_pct < 85.0:
        review_items.append(
            f"⚠️ <strong>Engineering Review Recommended:</strong> Model softmax confidence ({confidence_pct:.1f}%) "
            f"is below nominal 85.0% certainty guideline."
        )

    # 2. Secondary ambiguity flag
    if (confidence_pct - second_conf) < 25.0 and second_class:
        review_items.append(
            f"⚠️ <strong>Classification Ambiguity:</strong> High secondary probability detected for '<strong>{second_class}</strong>' "
            f"({second_conf:.1f}%). The wafer may exhibit overlapping or composite defect patterns."
        )

    # 3. Known sensitivity limitations for Scratch & Loc
    if pred_class == "Scratch":
        review_items.append(
            "ℹ️ <strong>Sensitivity Limitation:</strong> WM-811K test benchmark recall for Scratch is <strong>64.12%</strong> "
            "(22.94% escape rate to 'none'). Visual engineering verification of faint scratch tracks is recommended."
        )
    elif pred_class == "Loc":
        review_items.append(
            "ℹ️ <strong>Sensitivity Limitation:</strong> WM-811K test benchmark recall for Loc is <strong>77.97%</strong> "
            "(11.89% clean escape rate). Engineering review recommended for high-density die clusters."
        )

    if not review_items:
        review_items.append(
            "✅ <strong>Nominal Inspection Profile:</strong> Model prediction shows high certainty with no ambiguous secondary class. "
            "Proceed with standard manufacturing disposition workflow."
        )

    advisory_html = "".join(
        [f"<li style='margin-bottom: 6px; color: #1e293b;'>{item}</li>" for item in review_items]
    )

    st.markdown(
        f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin-top: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
            <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
                Review Advisories for Predicted Class: <span style="color: #0284c7;">{pred_class}</span> ({confidence_pct:.1f}% Confidence)
            </div>
            <ul style="padding-left: 20px; margin-bottom: 12px;">
                {advisory_html}
            </ul>
            <div style="font-size: 0.8rem; color: #64748b; border-top: 1px solid #f1f5f9; padding-top: 8px;">
                <strong>Engineering Scope Disclaimer:</strong> WaferVision AI predicts morphological defect categories based on spatial die patterns.
                It does not determine physical root-cause tool failures, wafer scrap decisions, or replace in-line electrical metrology.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
