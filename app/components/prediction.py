"""Prediction display component with probability charts and fab diagnostics."""

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_prediction_summary(result: dict[str, Any]) -> None:
    """Render prediction metrics, status cards, and physical mechanism details.

    Args:
        result: Dictionary output from WaferInferenceService.predict().
    """
    pred_class = result["predicted_class"]
    confidence_pct = result["confidence_pct"]
    is_clean = result["is_clean_wafer"]
    latency_ms = result["latency_ms"]
    device_name = result["device"]
    description = result.get("description", {})

    st.markdown("### 🎯 Classification Results")

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        status_label = "CLEAN WAFER" if is_clean else "DEFECT DETECTED"
        status_icon = "🟢" if is_clean else "🔴"
        st.markdown(
            f"""
            <div style="background-color: #121826; border-left: 5px solid {"#00ffc4" if is_clean else "#ff4b4b"}; padding: 15px; border-radius: 6px;">
                <span style="font-size: 0.85rem; color: #8e9bb0; text-transform: uppercase; letter-spacing: 1px;">{status_icon} Predicted Class</span>
                <h2 style="margin: 4px 0 0 0; color: #f0f4f8; font-size: 1.8rem;">{pred_class}</h2>
                <span style="color: {"#00ffc4" if is_clean else "#ff6b6b"}; font-weight: 600; font-size: 0.95rem;">{status_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color: #121826; padding: 15px; border-radius: 6px; border: 1px solid #1e293b;">
                <span style="font-size: 0.85rem; color: #8e9bb0; text-transform: uppercase; letter-spacing: 1px;">Prediction Confidence</span>
                <h2 style="margin: 4px 0 0 0; color: #38bdf8; font-size: 1.8rem;">{confidence_pct:.1f}%</h2>
                <span style="color: #64748b; font-size: 0.85rem;">Softmax probability</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color: #121826; padding: 15px; border-radius: 6px; border: 1px solid #1e293b;">
                <span style="font-size: 0.85rem; color: #8e9bb0; text-transform: uppercase; letter-spacing: 1px;">Inference Latency</span>
                <h2 style="margin: 4px 0 0 0; color: #a78bfa; font-size: 1.8rem;">{latency_ms:.2f} ms</h2>
                <span style="color: #64748b; font-size: 0.85rem;">Device: {device_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Human review notification if applicable
    if result.get("review_recommended", False):
        st.warning(
            "⚠️ **Human Review Recommended:** "
            + "; ".join(result.get("review_reasons", []))
            + ". High-volume fabrication standard operating procedure requires manual verification for borderline classifications."
        )

    # Physical fabrication mechanism context
    if description:
        with st.expander("🔬 Physical Fab Mechanism & Recommended Action", expanded=not is_clean):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Physical Mechanism:**\n{description.get('mechanism', 'N/A')}")
                st.markdown(f"**Geometric Morphology:**\n{description.get('morphology', 'N/A')}")
            with col_b:
                st.markdown(f"**Recommended Fab Action:**\n{description.get('fab_action', 'N/A')}")
                st.caption(
                    "Disclaimer: AI classification does not replace sub-micron metrology or destructive fault analysis."
                )


def render_probability_chart(probabilities: dict[str, float], predicted_class: str) -> None:
    """Render horizontal bar chart for complete 9-class probability distribution.

    Args:
        probabilities: Dictionary mapping class names to probabilities.
        predicted_class: Name of predicted class to highlight.
    """
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100.0 for c in classes]

    # Dynamic color palette
    colors = []
    for c in classes:
        if c == predicted_class:
            colors.append("#00d4ff" if predicted_class != "none" else "#00ffc4")
        else:
            colors.append("#1e293b")

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker={
                "color": colors,
                "line": {"color": "#334155", "width": 1},
            },
            text=[f"{p:.1f}%" if p >= 1.0 else "<1%" for p in probs],
            textposition="auto",
            textfont={"color": "#ffffff", "size": 11},
            hoverinfo="y+x",
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Class Probability Distribution (All 9 Canonical Defect Classes)",
            "font": {"size": 13, "color": "#f1f5f9"},
        },
        xaxis={
            "title": "Softmax Probability (%)",
            "range": [0, 105],
            "gridcolor": "#1e293b",
            "color": "#94a3b8",
            "zerolinecolor": "#334155",
        },
        yaxis={
            "autorange": "reversed",
            "color": "#f1f5f9",
            "tickfont": {"size": 12},
        },
        plot_bgcolor="rgba(10, 15, 29, 0.5)",
        paper_bgcolor="rgba(10, 15, 29, 0)",
        margin={"l": 80, "r": 20, "t": 35, "b": 35},
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)
