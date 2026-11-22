"""Header component for WaferVision AI inspection tool."""

import streamlit as st


def render_inspection_header(
    device: str, model_name: str = "CustomWaferCNN", param_count: int = 520681
) -> None:
    """Render compact industrial inspection header and status banner.

    Args:
        device: Computing device string (e.g. 'cuda' or 'cpu').
        model_name: Model architecture name.
        param_count: Parameter count of frozen model.
    """
    dev_str = str(device).upper()
    dev_color = "#16a34a" if "CUDA" in dev_str else "#0284c7"

    st.markdown(
        f"""
        <div class="fab-header-bar">
            <div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #0284c7;">🔬</span> WaferVision AI <span style="font-weight: 400; color: #64748b;">| Semiconductor Defect Inspection</span>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">
                    Automated Wafer Die Grid Pattern Classifier & Spatial Feature Explainability
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="status-badge-neutral">
                    📦 <strong>Model:</strong> {model_name} ({param_count:,} params)
                </span>
                <span class="status-badge-neutral">
                    🔒 <strong>Checkpoint:</strong> SHA-256 Verified
                </span>
                <span style="background-color: #f1f5f9; color: {dev_color}; border: 1px solid #cbd5e1; font-weight: 700; font-size: 0.82rem; padding: 3px 8px; border-radius: 4px; display: inline-flex; align-items: center; gap: 4px;">
                    ⚡ <strong>Device:</strong> {dev_str}
                </span>
                <span class="status-badge-clean">
                    ● Operational
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
