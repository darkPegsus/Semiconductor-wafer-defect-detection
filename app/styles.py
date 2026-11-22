"""Centralized CSS styles for WaferVision AI light industrial theme."""

LIGHT_INDUSTRIAL_CSS = """
<style>
    /* =========================================================================
       GLOBAL THEME & RESETS
       ========================================================================= */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Body Text & Paragraphs */
    p, span, label, div {
        color: #1e293b;
    }

    /* Captions & Secondary Text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569 !important;
        font-size: 0.85rem !important;
    }

    /* Horizontal Rules */
    hr {
        border-color: #e2e8f0 !important;
        margin: 1.2rem 0 !important;
    }

    /* =========================================================================
       SIDEBAR NAVIGATION & STATUS
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #1e293b !important;
    }

    /* Sidebar Radio Navigation items */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        transition: all 0.15s ease-in-out;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #e0f2fe !important;
        border-color: #38bdf8 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] span {
        color: #0369a1 !important;
        font-weight: 700 !important;
    }

    /* =========================================================================
       CARDS & CONTAINERS
       ========================================================================= */
    .fab-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .fab-card-compact {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    .fab-header-bar {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }

    /* =========================================================================
       METRIC CARDS
       ========================================================================= */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0284c7 !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }

    /* =========================================================================
       BUTTONS
       ========================================================================= */
    /* Primary Action Button (ANALYZE WAFER) */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.25) !important;
        letter-spacing: 0.03em !important;
        transition: all 0.15s ease-in-out !important;
    }

    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #0369a1 !important;
        box-shadow: 0 4px 6px rgba(2, 132, 199, 0.35) !important;
    }

    /* Secondary Button */
    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* =========================================================================
       INPUT CONTROLS (Selectboxes, File Uploader, Radio, Sliders)
       ========================================================================= */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] * {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    /* File Uploader Dropzone */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #94a3b8 !important;
        border-radius: 8px !important;
        padding: 24px 16px !important;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #0284c7 !important;
        background-color: #f0f9ff !important;
    }

    section[data-testid="stFileUploadDropzone"] span,
    section[data-testid="stFileUploadDropzone"] small {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* Slider Track & Thumb */
    div[data-testid="stSlider"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* =========================================================================
       STATUS BADGES & INSPECTION TAGS
       ========================================================================= */
    .status-badge-clean {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        letter-spacing: 0.02em;
    }

    .status-badge-defect {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        letter-spacing: 0.02em;
    }

    .status-badge-neutral {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .status-badge-accent {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* =========================================================================
       DATAFRAME & TABLE STYLING
       ========================================================================= */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
</style>
"""
