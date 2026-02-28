import streamlit as st
import pandas as pd
import requests
import hashlib
import os

from styles import (
    CSS,
    section_label, doc_card, health_indicator,
    confidence_bar, escalation_warning,
    query_row, gap_row
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FDE Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS, unsafe_allow_html=True)


# Helpers 

def generate_user_id(workspace: str, passkey: str) -> str:
    raw = f"{workspace.strip().lower()}:{passkey.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

def api_get(path: str, **params):
    params["user_id"] = st.session_state.get("user_id", "default")
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(path: str, json: dict = None, files=None, **params):
    params["user_id"] = st.session_state.get("user_id", "default")
    if json is not None:
        json["user_id"] = st.session_state.get("user_id", "default")
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            params=params if files else None,
            json=json,
            files=files,
            timeout=30
        )
        return r.json() if r.status_code == 200 else r.json()
    except Exception as e:
        return {"detail": str(e)}


# Session state 
defaults = {
    "messages":   [],
    "session_id": None,
    "model":      "llama-3.1-8b-instant",
    "documents":  None,
    "analytics":  None,
    "audit":      None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# Workspace login 
if "user_id" not in st.session_state:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="padding:60px 0 32px 0;text-align:center;">
            <div style="font-size:20px;font-weight:600;color:#e8e8f0;margin-bottom:8px;">
                FDE Assistant
            </div>
            <div style="font-size:13px;color:#55556a;line-height:1.7;">
                Enter your workspace name and passkey to continue.<br>
                First time? Just pick any name and passkey to create one.
            </div>
        </div>
        """, unsafe_allow_html=True)

        workspace = st.text_input("Workspace", placeholder="e.g. acme-team")
        passkey   = st.text_input("Passkey", type="password",
                                   placeholder="Your secret key")

        if st.button("Continue", use_container_width=True):
            if workspace and passkey:
                st.session_state.user_id   = generate_user_id(workspace, passkey)
                st.session_state.workspace = workspace
                st.rerun()
            else:
                st.error("Both fields are required.")
    st.stop()


# Header

workspace_label = st.session_state.get("workspace", "")
st.markdown(f"""
<div style="padding:8px 0 20px 0;border-bottom:1px solid #2a2a3a;margin-bottom:20px;">
    <div style="font-size:17px;font-weight:600;color:#e8e8f0;">
        FDE Assistant
    </div>
    <div style="font-size:12px;color:#55556a;margin-top:2px;">
        {workspace_label}
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar

with st.sidebar:

    st.markdown(section_label("Model"), unsafe_allow_html=True)
    selected = st.selectbox("Model", ["llama-3.1-8b-instant"],
                             label_visibility="collapsed")
    st.session_state.model = selected

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(section_label("Upload document"), unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf", "docx", "html"],
                                      label_visibility="collapsed")
    if uploaded_file:
        if st.button("Upload and index", use_container_width=True):
            with st.spinner("Indexing..."):
                files = {"file": (uploaded_file.name,
                                  uploaded_file,
                                  uploaded_file.type)}
                resp = api_post("/upload-doc", files=files)
                if resp and resp.get("file_id"):
                    st.success("Done")
                    st.session_state.documents = None
                else:
                    st.error(resp.get("detail", "Upload failed"))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(section_label("Documents"), unsafe_allow_html=True)

    if st.button("Refresh", use_container_width=True):
        st.session_state.documents = None

    if st.session_state.documents is None:
        st.session_state.documents = api_get("/list-docs") or []

    docs = st.session_state.documents
    if docs:
        for doc in docs:
            st.markdown(doc_card(doc["filename"]), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sel_id = st.selectbox(
            "Select to delete",
            options=[d["id"] for d in docs],
            format_func=lambda x: next(
                d["filename"] for d in docs if d["id"] == x
            ),
            label_visibility="collapsed"
        )
        if st.button("Delete selected", use_container_width=True):
            with st.spinner("Deleting..."):
                resp = api_post("/delete-doc", json={"file_id": sel_id})
                if resp and "message" in resp:
                    st.success("Deleted")
                    st.session_state.documents = None
                else:
                    st.error(resp.get("detail", "Delete failed"))
    else:
        st.markdown("""
        <div style="padding:16px 0;color:#55556a;font-size:12px;">
            No documents yet
        </div>
        """, unsafe_allow_html=True)

    # API health
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
        ok     = health.get("status") == "healthy"
        color, label = ("#22c55e", "API healthy") if ok else ("#f59e0b", "API degraded")
    except Exception:
        color, label = "#ef4444", "API offline"

    st.markdown(health_indicator(color, label), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Switch workspace", use_container_width=True):
        for key in ["user_id", "workspace", "messages", "session_id",
                    "documents", "analytics", "audit"]:
            st.session_state.pop(key, None)
        st.rerun()


# Tabs 
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Bulk query", "Analytics", "Audit log"])


# Chat
with tab1:

    with st.expander("Delivery options", expanded=False):
        notify_slack = st.checkbox("Send answer to Slack", key="notify_slack")
        if notify_slack:
            try:
                health   = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
                slack_ok = health.get("checks", {}).get("slack") == "configured"
                if slack_ok:
                    st.success("Slack is configured")
                else:
                    st.warning("Slack not configured. Add SLACK_WEBHOOK_URL to .env")
            except Exception:
                st.caption("Could not check Slack status.")

    if not st.session_state.messages:
        st.markdown("""
        <div style="padding:60px 0;text-align:center;color:#55556a;">
            <div style="font-size:14px;color:#8888aa;margin-bottom:6px;">
                Ask anything about your uploaded documents
            </div>
            <div style="font-size:12px;">
                Upload a PDF, DOCX, or HTML file to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                st.markdown(
                    confidence_bar(meta.get("confidence", 0),
                                   meta.get("sources", [])),
                    unsafe_allow_html=True
                )
                if meta.get("escalated"):
                    st.markdown(escalation_warning(), unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(""):
                resp = api_post("/chat", json={
                    "question":     prompt,
                    "session_id":   st.session_state.session_id,
                    "model":        st.session_state.model,
                    "notify_slack": st.session_state.get("notify_slack", False),
                })
                if resp and "answer" in resp:
                    st.session_state.session_id = resp.get("session_id")
                    st.markdown(resp["answer"])
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": resp["answer"],
                        "meta": {
                            "confidence": resp.get("confidence", 0),
                            "sources":    resp.get("sources", []),
                            "escalated":  resp.get("escalated", False),
                        }
                    })
                    st.rerun()
                else:
                    err = resp.get("detail", "Unknown error") if resp else "No response"
                    st.error(f"Error: {err}")


# Bulk query 
with tab2:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:14px;font-weight:500;color:#e8e8f0;margin-bottom:6px;">
            Bulk questionnaire
        </div>
        <div style="font-size:13px;color:#8888aa;line-height:1.6;">
            Upload a CSV with a <code style="background:#16161f;padding:2px 6px;
            border-radius:4px;font-size:11px;">question</code> column.
            Every question gets answered and flagged by confidence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    qfile = st.file_uploader("Upload CSV", type=["csv"], key="bulk_upload",
                              label_visibility="collapsed")
    if qfile:
        if st.button("Answer all questions"):
            with st.spinner("Processing..."):
                files  = {"file": (qfile.name, qfile, "text/csv")}
                result = api_post("/answer-questionnaire", files=files)

                if result and "results" in result:
                    total  = result["total"]
                    review = result["needs_review_count"]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total",         total)
                    col2.metric("Auto-answered",  total - review)
                    col3.metric("Needs review",   review)

                    df = pd.DataFrame(result["results"])
                    df["confidence"]  = df["confidence"].apply(lambda x: f"{int(x*100)}%")
                    df["needs_review"] = df["needs_review"].map(
                        {True: "Review", False: "OK"}
                    )
                    df["sources"] = df["sources"].apply(
                        lambda x: ", ".join(x) if x else ""
                    )
                    st.dataframe(
                        df[["question", "answer", "confidence",
                            "needs_review", "sources"]],
                        use_container_width=True
                    )
                    st.download_button(
                        "Download answers",
                        df.to_csv(index=False),
                        "answered_questionnaire.csv",
                        "text/csv"
                    )
                else:
                    err = result.get("detail", "Error") if result else "No response"
                    st.error(err)
    else:
        st.markdown("""
        <div style="padding:32px;text-align:center;
                    border:1px dashed #2a2a3a;border-radius:8px;
                    color:#55556a;font-size:13px;">
            Upload a CSV file with a question column
        </div>
        """, unsafe_allow_html=True)


# Analytics 
with tab3:
    if st.button("Refresh"):
        st.session_state.analytics = None

    if st.session_state.analytics is None:
        st.session_state.analytics = api_get("/analytics")

    data = st.session_state.analytics
    if data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total queries",   data["total_queries"])
        col2.metric("Escalated",       data["escalated_count"])
        col3.metric("Avg confidence",  f"{int(data['avg_confidence'] * 100)}%")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(section_label("Recent queries"), unsafe_allow_html=True)
            for i, q in enumerate(data["top_questions"]):
                st.markdown(query_row(i + 1, q), unsafe_allow_html=True)

        with col2:
            st.markdown(section_label("Low confidence queries"), unsafe_allow_html=True)
            if data["unanswered_topics"]:
                for q in data["unanswered_topics"]:
                    st.markdown(gap_row(q), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="padding:16px 0;color:#22c55e;font-size:13px;">
                    No gaps detected
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No data yet. Ask some questions first.")


# Audit log 
with tab4:
    if st.button("Refresh", key="refresh_audit"):
        st.session_state.audit = None

    if st.session_state.audit is None:
        st.session_state.audit = api_get("/audit-log") or []

    audit_data = st.session_state.audit
    if audit_data:
        df = pd.DataFrame(audit_data)
        df["escalated"]  = df["escalated"].map({0: "OK", 1: "Review"})
        df["confidence"] = df["confidence"].apply(lambda x: f"{int(x*100)}%")
        st.dataframe(
            df[["created_at", "user_query", "confidence", "escalated", "sources"]],
            use_container_width=True
        )
        st.download_button(
            "Export as CSV",
            df.to_csv(index=False),
            "audit_log.csv",
            "text/csv"
        )
    else:
        st.info("No queries logged yet.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(section_label("System logs"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    log_level = col1.selectbox("Level", ["all", "ERROR", "WARNING", "INFO"])
    log_limit = col2.number_input("Limit", value=50, min_value=10, max_value=500)

    if st.button("Load logs"):
        params = {"limit": int(log_limit)}
        if log_level != "all":
            params["level"] = log_level
        result = api_get("/logs", **params)

        if result and result.get("logs"):
            df_logs = pd.DataFrame(result["logs"])
            st.dataframe(df_logs, use_container_width=True)
            st.download_button(
                "Export logs",
                df_logs.to_csv(index=False),
                "system_logs.csv",
                "text/csv"
            )
        else:
            st.info("No logs found.")