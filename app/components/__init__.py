"""Streamlit UI components for WaferVision AI."""

from app.components.about import render_about_page
from app.components.dashboard import render_performance_dashboard
from app.components.header import render_inspection_header
from app.components.prediction import (
    render_inspection_notes,
    render_prediction_summary,
    render_probability_chart,
)
from app.components.upload import render_input_selector
from app.components.visualization import render_explainability_panel

__all__ = [
    "render_about_page",
    "render_performance_dashboard",
    "render_inspection_header",
    "render_prediction_summary",
    "render_probability_chart",
    "render_inspection_notes",
    "render_input_selector",
    "render_explainability_panel",
]
