FONTS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
"""

VARIABLES = """
:root {
    --bg-primary:     #0a0a0f;
    --bg-secondary:   #111118;
    --bg-card:        #16161f;
    --bg-hover:       #1e1e2a;
    --border:         #2a2a3a;
    --border-bright:  #3a3a52;
    --accent:         #4f6ef7;
    --accent-dim:     #2a3a8a;
    --accent-glow:    rgba(79,110,247,0.15);
    --green:          #22c55e;
    --green-dim:      rgba(34,197,94,0.1);
    --yellow:         #f59e0b;
    --yellow-dim:     rgba(245,158,11,0.1);
    --red:            #ef4444;
    --red-dim:        rgba(239,68,68,0.1);
    --text-primary:   #e8e8f0;
    --text-secondary: #8888aa;
    --text-muted:     #55556a;
    --font-main:      'IBM Plex Sans', sans-serif;
    --font-mono:      'IBM Plex Mono', monospace;
}
"""

GLOBAL = """
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background-color: var(--bg-primary) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
"""

SIDEBAR = """
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-glow) !important;
}
"""

TABS = """
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-family: var(--font-mono) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-hover) !important;
}
"""

CHAT = """
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-bright) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
}
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}
"""

COMPONENTS = """
/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3d5ce6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,110,247,0.3) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-secondary) !important;
    font-size: 12px !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-family: var(--font-mono) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-size: 13px !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 20px 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* Password input */
[data-testid="stPasswordInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* Status messages */
.stSuccess {
    background: var(--green-dim) !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
}
.stError {
    background: var(--red-dim) !important;
    border: 1px solid var(--red) !important;
}
.stWarning {
    background: var(--yellow-dim) !important;
    border: 1px solid var(--yellow) !important;
}
"""

# ── Assembled stylesheet ───────────────────────────────────────────────────────
CSS = f"<style>{FONTS}{VARIABLES}{GLOBAL}{SIDEBAR}{TABS}{CHAT}{COMPONENTS}</style>"


# ── Reusable HTML snippets ─────────────────────────────────────────────────────

def section_label(text: str) -> str:
    """Simple muted label for sidebar and tab sections."""
    return f"""
    <div style="font-size:11px;color:#55556a;margin-bottom:8px;">
        {text}
    </div>
    """

def doc_card(filename: str) -> str:
    """Document card in sidebar list."""
    return f"""
    <div style="padding:8px 10px;margin:4px 0;background:#16161f;
                border:1px solid #2a2a3a;border-radius:6px;
                font-size:12px;color:#e8e8f0;
                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
        {filename}
    </div>
    """

def health_indicator(color: str, label: str) -> str:
    """API health status shown at bottom of sidebar."""
    return f"""
    <div style="padding:8px 10px;background:#16161f;border:1px solid #2a2a3a;
                border-radius:6px;display:flex;align-items:center;gap:8px;
                font-size:12px;">
        <div style="width:6px;height:6px;background:{color};
                    border-radius:50%;flex-shrink:0;"></div>
        <span style="color:#55556a;">{label}</span>
    </div>
    """

def confidence_bar(conf: float, sources: list) -> str:
    """Confidence bar and source chips below each assistant message."""
    pct = int(conf * 100)
    if conf > 0.7:
        color, label = "#22c55e", "High"
    elif conf > 0.4:
        color, label = "#f59e0b", "Medium"
    else:
        color, label = "#ef4444", "Low"

    src_html = ""
    if sources:
        chips = "".join([
            f'<span style="background:#1e1e2a;border:1px solid #2a2a3a;'
            f'border-radius:4px;padding:2px 8px;font-size:11px;'
            f'color:#8888aa;">{s}</span>'
            for s in sources
        ])
        src_html = f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px;">{chips}</div>'

    return f"""
    <div style="margin-top:8px;padding:10px 12px;background:#111118;
                border:1px solid #2a2a3a;border-radius:6px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <span style="font-size:12px;color:{color};">Confidence: {pct}% ({label})</span>
            <div style="flex:1;height:2px;background:#2a2a3a;border-radius:2px;">
                <div style="width:{pct}%;height:2px;background:{color};border-radius:2px;"></div>
            </div>
        </div>
        {src_html}
    </div>
    """

def escalation_warning() -> str:
    return """
    <div style="margin-top:6px;padding:8px 12px;
                background:rgba(245,158,11,0.06);
                border-left:3px solid #f59e0b;border-radius:4px;
                font-size:12px;color:#f59e0b;">
        Low confidence — verify before sharing with the customer.
    </div>
    """

def query_row(index: int, text: str) -> str:
    return f"""
    <div style="padding:8px 12px;margin:4px 0;background:#16161f;
                border:1px solid #2a2a3a;border-radius:6px;
                font-size:12px;color:#8888aa;display:flex;gap:10px;">
        <span style="color:#3a3a52;min-width:20px;">{index}.</span>
        <span style="color:#e8e8f0;">{text}</span>
    </div>
    """

def gap_row(text: str) -> str:
    return f"""
    <div style="padding:8px 12px;margin:4px 0;
                background:rgba(239,68,68,0.04);
                border-left:3px solid #ef4444;border-radius:4px;
                font-size:12px;color:#8888aa;">
        {text}
    </div>
    """