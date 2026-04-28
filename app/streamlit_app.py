import streamlit as st
import pandas as pd
import requests
import os
import html as _html

from styles import (
    CSS,
    section_label, doc_card, health_indicator,
    escalation_warning,
    query_row, gap_row
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@st.cache_data(ttl=15)
def get_health_status():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=2).json()
    except Exception:
        return None

st.set_page_config(
    page_title="FDE Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers() -> dict:
    headers = {}
    token = st.session_state.get("auth_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api_key = os.getenv("API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def api_get(path: str, **params):
    # Tenant identity is carried in the Authorization Bearer header (see _headers).
    # We no longer add a `user_id` query param — the API ignores it and the old
    # spoofable fallback has been removed.
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params or None,
                         headers=_headers(), timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(path: str, json_body: dict = None, files=None, **params):
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            params=params if files else None,
            json=json_body,
            files=files,
            headers=_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        try:
            return r.json()
        except Exception:
            return {"detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"detail": str(e)}


# ── Session state ─────────────────────────────────────────────────────────────

defaults = {
    "brief":      None,
    "documents":  None,
    "analytics":  None,
    "audit":      None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Workspace login ───────────────────────────────────────────────────────────

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
        passkey   = st.text_input("Passkey", type="password", placeholder="Your secret key")

        if st.button("Continue", use_container_width=True):
            if workspace and passkey:
                # We must obtain a real JWT here. The previous version silently
                # fell back to a locally-hashed user_id when /auth/token failed,
                # which let backend outages / wrong passkeys (401s) appear as
                # successful logins and downgraded the session to unauthenticated
                # API calls. The API now requires a Bearer token, so we surface
                # the failure instead of degrading.
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/auth/token",
                        json={"workspace": workspace, "passkey": passkey},
                        timeout=5,
                    )
                except requests.RequestException as e:
                    st.error(f"Auth service unreachable: {e}")
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.auth_token = data["token"]
                        st.session_state.user_id = data["user_id"]
                        st.session_state.workspace = workspace
                        st.rerun()
                    elif resp.status_code == 429:
                        st.error("Too many login attempts. Please wait a minute and try again.")
                    else:
                        try:
                            detail = resp.json().get("detail", resp.text[:120])
                        except Exception:
                            detail = resp.text[:120] or f"HTTP {resp.status_code}"
                        st.error(f"Login failed: {detail}")
            else:
                st.error("Both fields are required.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────

workspace_label = st.session_state.get("workspace", "")
st.markdown(f"""
<div style="padding:8px 0 20px 0;border-bottom:1px solid #2a2a3a;margin-bottom:20px;">
    <div style="font-size:17px;font-weight:600;color:#e8e8f0;">
        FDE Assistant
    </div>
    <div style="font-size:12px;color:#55556a;margin-top:2px;">
        {_html.escape(workspace_label)}
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(section_label("Upload document"), unsafe_allow_html=True)

    doc_type = st.selectbox(
        "Document type",
        ["auto", "pdf", "transcript", "ticket"],
        help="'auto' infers from file extension. Select explicitly for .txt and .json files.",
    )

    uploaded_file = st.file_uploader(
        "",
        type=["pdf", "docx", "html", "txt", "json"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        if st.button("Upload and index", use_container_width=True):
            with st.spinner("Indexing..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                resp = requests.post(
                    f"{API_BASE_URL}/upload-doc",
                    files=files,
                    params={"doc_type": doc_type},
                    headers=_headers(),
                    timeout=60,
                )
                data = resp.json() if resp.content else {}
                if resp.status_code == 200 and data.get("file_id"):
                    st.success("Done")
                    st.session_state.documents = None
                else:
                    st.error(data.get("detail", "Upload failed"))

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
            format_func=lambda x: next(d["filename"] for d in docs if d["id"] == x),
            label_visibility="collapsed",
        )
        if st.button("Delete selected", use_container_width=True):
            with st.spinner("Deleting..."):
                resp = api_post("/delete-doc", json_body={"file_id": sel_id})
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

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    health = get_health_status()
    if health:
        ok = health.get("status") == "healthy"
        color, label = ("#22c55e", "API healthy") if ok else ("#f59e0b", "API degraded")
    else:
        color, label = "#ef4444", "API offline"

    st.markdown(health_indicator(color, label), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Switch workspace", use_container_width=True):
        for key in ["user_id", "workspace", "auth_token", "brief",
                    "documents", "analytics", "audit"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Brief", "Bulk query", "Analytics", "Audit log"])


# ── Brief viewer ──────────────────────────────────────────────────────────────

with tab1:
    st.markdown("""
    <div style="margin-bottom:16px;">
        <div style="font-size:14px;font-weight:500;color:#e8e8f0;margin-bottom:4px;">
            Pre-call brief
        </div>
        <div style="font-size:12px;color:#8888aa;">
            Describe what you need to know before your customer call.
            The assistant will surface issues, risks, and talking points from your documents.
        </div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Query",
        placeholder="e.g. What are the open issues and risks for this customer?",
        label_visibility="collapsed",
    )

    if st.button("Generate brief", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            with st.spinner("Analyzing documents..."):
                result = api_post("/brief", json_body={
                    "query": query,
                    "customer_id": st.session_state.get("user_id"),
                })
                if result and "brief" in result:
                    st.session_state.brief = result
                else:
                    st.error(result.get("detail", "Brief generation failed."))

    # ── Render saved brief ─────────────────────────────────────────────────────
    if st.session_state.brief:
        result = st.session_state.brief
        brief = result.get("brief", {})
        faithfulness = result.get("faithfulness_score", 0.0)
        loop_count = result.get("loop_count", 0)

        # Header metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", f"{int(faithfulness * 100)}%")
        if brief.get("judge_status") == "ok":
            col1.markdown(
                "<span style='display:inline-block;padding:2px 8px;background:#14331f;"
                "color:#6ee7b7;border:1px solid #2f7a52;border-radius:10px;"
                "font-size:11px;letter-spacing:0.3px;'>Verified</span>",
                unsafe_allow_html=True,
            )
        col2.metric("Issues + Risks",
                    len(brief.get("issues", [])) + len(brief.get("risks", [])))
        col3.metric("Retrieval loops", loop_count)

        # Only render a banner when verification didn't run cleanly — stays out of
        # the way in the happy path so there's nothing extra for users to parse.
        _JUDGE_MESSAGES = {
            "skipped_breaker_open": (
                "amber",
                "Verification temporarily unavailable — claims have not been "
                "independently checked. Try again in a few minutes.",
            ),
            "error": (
                "amber",
                "Verification step failed — claims have not been independently "
                "checked. Review citations before acting.",
            ),
            "parse_error": (
                "amber",
                "Verification step failed — claims have not been independently "
                "checked. Review citations before acting.",
            ),
            "no_context_all_unsupported": (
                "red",
                "No supporting documents found for this query. This brief is not "
                "grounded in your uploaded docs.",
            ),
        }
        _msg = _JUDGE_MESSAGES.get(brief.get("judge_status"))
        if _msg:
            _color, _text = _msg
            _bg = "#3a2a10" if _color == "amber" else "#3a1010"
            _border = "#d19a3a" if _color == "amber" else "#d14a4a"
            st.markdown(
                f"<div style='padding:10px 14px;background:{_bg};border-left:3px solid "
                f"{_border};border-radius:6px;margin:12px 0;color:#e8e8e8;font-size:13px;'>"
                f"{_html.escape(_text)}</div>",
                unsafe_allow_html=True,
            )

        if faithfulness < 0.4:
            st.markdown(escalation_warning(), unsafe_allow_html=True)

        # Summary
        if brief.get("summary"):
            st.markdown(f"""
            <div style="padding:12px 16px;background:#16161f;border-radius:8px;
                        border-left:3px solid #6366f1;margin:16px 0;
                        color:#c8c8e0;font-size:13px;">
                {_html.escape(brief['summary'])}
            </div>
            """, unsafe_allow_html=True)

        # Issues
        if brief.get("issues"):
            st.markdown(section_label("Issues"), unsafe_allow_html=True)
            for item in brief["issues"]:
                with st.expander(item.get("claim", ""), expanded=False):
                    if item.get("source_doc"):
                        st.caption(f"Source: {item['source_doc']}")
                    if item.get("passage"):
                        st.markdown(f"> {item['passage']}")

        # Risks
        if brief.get("risks"):
            st.markdown(section_label("Risks"), unsafe_allow_html=True)
            for item in brief["risks"]:
                with st.expander(item.get("claim", ""), expanded=False):
                    if item.get("source_doc"):
                        st.caption(f"Source: {item['source_doc']}")
                    if item.get("passage"):
                        st.markdown(f"> {item['passage']}")

        # Talking points
        if brief.get("talking_points"):
            st.markdown(section_label("Talking points"), unsafe_allow_html=True)
            for item in brief["talking_points"]:
                with st.expander(item.get("point", ""), expanded=False):
                    if item.get("source_doc"):
                        st.caption(f"Source: {item['source_doc']}")
                    if item.get("passage"):
                        st.markdown(f"> {item['passage']}")

        # Open questions
        if brief.get("open_questions"):
            st.markdown(section_label("Open questions"), unsafe_allow_html=True)
            for q in brief["open_questions"]:
                st.markdown(f"- {_html.escape(q)}")

        # Information gaps
        if brief.get("information_gaps"):
            st.markdown(section_label("Information gaps"), unsafe_allow_html=True)
            for g in brief["information_gaps"]:
                st.markdown(f"- {_html.escape(g)}")

        # Sources
        sources = result.get("sources") or brief.get("sources", [])
        if sources:
            st.markdown(section_label("Sources"), unsafe_allow_html=True)
            for s in sources:
                label = s.get("filename", "unknown")
                if s.get("doc_type"):
                    label += f" ({s['doc_type']})"
                st.markdown(doc_card(label), unsafe_allow_html=True)

        # Suspicious facts — regex-detected atomic fact mismatches
        if brief.get("suspicious_facts"):
            with st.expander("Potentially ungrounded facts", expanded=False):
                for f in brief["suspicious_facts"]:
                    st.markdown(f"- `{f}`")

        # Suspicious claims — claim-level verdicts from regex + LLM judge
        if brief.get("suspicious_claims"):
            sc_list = brief["suspicious_claims"]
            with st.expander(f"Hallucination warnings — {len(sc_list)} claim(s) flagged", expanded=True):
                for sc in sc_list:
                    layer = "LLM judge" if sc.get("caught_by") == "llm_judge" else "Regex"
                    st.markdown(
                        f"**[{layer}]** {_html.escape(sc.get('claim', ''))}"
                    )
                    st.caption(sc.get("reason", ""))

    else:
        st.markdown("""
        <div style="padding:60px 0;text-align:center;color:#55556a;">
            <div style="font-size:14px;color:#8888aa;margin-bottom:6px;">
                Enter a query above to generate your pre-call brief
            </div>
            <div style="font-size:12px;">
                Upload PDFs, transcripts, or tickets to get started
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Bulk query ────────────────────────────────────────────────────────────────

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
            Maximum 200 rows per batch.
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
                    col1.metric("Total",        total)
                    col2.metric("Auto-answered", total - review)
                    col3.metric("Needs review",  review)

                    df = pd.DataFrame(result["results"])
                    df["confidence"]   = df["confidence"].apply(lambda x: f"{int(x*100)}%")
                    df["needs_review"] = df["needs_review"].map({True: "Review", False: "OK"})
                    df["sources"]      = df["sources"].apply(
                        lambda x: ", ".join(x) if x else ""
                    )
                    st.dataframe(
                        df[["question", "answer", "confidence", "needs_review", "sources"]],
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download answers",
                        df.to_csv(index=False),
                        "answered_questionnaire.csv",
                        "text/csv",
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


# ── Analytics ─────────────────────────────────────────────────────────────────

with tab3:
    if st.button("Refresh"):
        st.session_state.analytics = None

    if st.session_state.analytics is None:
        st.session_state.analytics = api_get("/analytics")

    data = st.session_state.analytics
    if data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total queries",  data["total_queries"])
        col2.metric("Escalated",      data["escalated_count"])
        col3.metric("Avg confidence", f"{int(data['avg_confidence'] * 100)}%")

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
        st.info("No data yet. Generate some briefs first.")


# ── Audit log ─────────────────────────────────────────────────────────────────

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
            use_container_width=True,
        )
        st.download_button(
            "Export as CSV",
            df.to_csv(index=False),
            "audit_log.csv",
            "text/csv",
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
                "text/csv",
            )
        else:
            st.info("No logs found.")
