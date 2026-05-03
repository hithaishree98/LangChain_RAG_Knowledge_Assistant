import streamlit as st
import os
import requests
import html as _html
import pandas as pd

from styles import load_css

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="FDE Assistant", page_icon="📋", layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)

# ── API helpers ───────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
API_KEY = os.getenv("API_KEY", "")
# Demo workspace credentials — override via env for non-demo deployments.
_DEMO_WORKSPACE = os.getenv("DEMO_WORKSPACE", "demo")
_DEMO_PASSKEY   = os.getenv("DEMO_PASSKEY",   "demo")


def _get_auth_token() -> str:
    """Return a cached JWT for the demo workspace, minting one on first call."""
    if "auth_token" not in st.session_state:
        try:
            r = requests.post(
                f"{API_URL}/auth/token",
                json={"workspace": _DEMO_WORKSPACE, "passkey": _DEMO_PASSKEY},
                timeout=5,
            )
            st.session_state.auth_token = r.json().get("token", "") if r.ok else ""
        except Exception:
            st.session_state.auth_token = ""
    return st.session_state.get("auth_token", "")


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    token = _get_auth_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def api_get(path, **kwargs):
    try:
        r = requests.get(f"{API_URL}{path}", headers=_headers(), timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def api_post(path, json_body=None, **kwargs):
    try:
        r = requests.post(
            f"{API_URL}{path}", headers=_headers(), json=json_body, timeout=120, **kwargs
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def api_post_files(path, files, data=None):
    """Post multipart/form-data (file upload). Does not set Content-Type — requests sets it."""
    h = {}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    token = _get_auth_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(
            f"{API_URL}{path}", headers=h, files=files, data=data or {}, timeout=600
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Session state defaults ────────────────────────────────────────────────────

if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None
if "customers" not in st.session_state:
    st.session_state.customers = []
if "pre_brief" not in st.session_state:
    st.session_state.pre_brief = None
if "exec_brief" not in st.session_state:
    st.session_state.exec_brief = None
if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "corpus_health" not in st.session_state:
    st.session_state.corpus_health = None


# ── Display helpers ───────────────────────────────────────────────────────────

def render_commitment(c: dict):
    if c.get("is_overdue"):
        status_color = "🔴"
    elif c.get("is_slipped"):
        status_color = "🟡"
    else:
        status_color = "🟢"
    st.markdown(f"{status_color} **{c.get('description', '')}**")
    cols = st.columns(3)
    if c.get("target_date"):
        cols[0].caption(f"Target: {c['target_date']}")
    if c.get("owner"):
        cols[1].caption(f"Owner: {c['owner']}")
    cols[2].caption(f"Status: {c.get('status', 'open')}")
    src = c.get("source") or {}
    if isinstance(src, dict) and src.get("is_stale"):
        st.warning("Source may be stale")


def render_open_item(item: dict):
    priority = item.get("priority", "normal")
    badge = {
        "P0": "🔴 P0",
        "P1": "🟠 P1",
        "P2": "🟡 P2",
        "CRITICAL": "🔴 CRIT",
        "BLOCKER": "🔴 BLOCKER",
    }.get(priority.upper(), f"⚪ {priority}")
    verification = item.get("verification") or {}
    flag = verification.get("flag") if isinstance(verification, dict) else None
    flag_str = {
        "stale_source": " ⚠️ stale",
        "verify_before_quoting": " 🔍 verify",
        "conflict": " ⚡ conflict",
    }.get(flag, "")
    st.markdown(f"{badge} **{item.get('title', '')}**{flag_str}")
    st.caption(
        f"Status: {item.get('status', '')} | "
        f"Owner: {item.get('owner', '')} | "
        f"Updated: {item.get('last_update', 'unknown')}"
    )


def section_status_label(status: str) -> str:
    """Return an HTML label for section_status values."""
    if status == "unavailable":
        return '<span style="color:#888;font-size:12px;">No data available</span>'
    if status == "empty":
        return '<span style="color:#aaa;font-size:12px;font-style:italic;">No items found</span>'
    return ""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### FDE Assistant")
    st.caption("Single-workspace demo — all data is scoped to one API key.")
    st.markdown("---")

    if st.button("Refresh customers", use_container_width=True):
        st.session_state.customers = []
        st.session_state.corpus_health = None

    if not st.session_state.customers:
        with st.spinner("Loading customers..."):
            result = api_get("/customers")
            if "error" in (result or {}):
                st.error(f"Could not load customers: {result['error']}")
            else:
                st.session_state.customers = result if isinstance(result, list) else []

    customers = st.session_state.customers

    if customers:
        customer_names = [c["name"] for c in customers]
        current_idx = 0
        if st.session_state.selected_customer:
            try:
                current_idx = next(
                    i for i, c in enumerate(customers)
                    if c["id"] == st.session_state.selected_customer["id"]
                )
            except StopIteration:
                current_idx = 0

        selected_name = st.selectbox(
            "Select customer",
            options=customer_names,
            index=current_idx,
        )
        selected = next((c for c in customers if c["name"] == selected_name), None)
        if selected and selected != st.session_state.selected_customer:
            st.session_state.selected_customer = selected
            st.session_state.corpus_health = None
            st.session_state.pre_brief = None
            st.session_state.exec_brief = None
            st.session_state.query_result = None
    else:
        st.info("No customers yet. Create one below.")

    if st.session_state.selected_customer:
        c = st.session_state.selected_customer
        st.markdown(
            f'<div style="padding:8px;background:#16161f;border:1px solid #2a2a3a;'
            f'border-radius:6px;margin:8px 0;">'
            f'<div style="font-size:13px;font-weight:600;color:#e8e8f0;">{_html.escape(c["name"])}</div>'
            f'<div style="font-size:11px;color:#55556a;">{_html.escape(c.get("slug",""))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.corpus_health is None:
            health = api_get(f"/customers/{c['slug']}/corpus-health")
            if "error" not in (health or {}):
                st.session_state.corpus_health = health

        ch = st.session_state.corpus_health or {}
        if ch and "error" not in ch:
            st.markdown("**Corpus Health**")
            doc_types = ch.get("doc_types") or {}
            if doc_types:
                for dtype, info in doc_types.items():
                    count = info.get("count", 0) if isinstance(info, dict) else info
                    st.markdown(
                        f'<div style="font-size:11px;color:#8888aa;padding:2px 0;">'
                        f'{_html.escape(str(dtype))}: {count} doc(s)</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-size:11px;color:#55556a;">No documents indexed</div>',
                    unsafe_allow_html=True,
                )
            last_call = ch.get("last_call_date")
            if last_call:
                st.caption(f"Last call: {last_call}")
            overall = ch.get("overall", "")
            if overall:
                color = "#22c55e" if overall == "current" else "#f59e0b" if overall == "stale" else "#ef4444"
                st.markdown(
                    f'<div style="font-size:11px;color:{color};margin-top:4px;">'
                    f'Corpus: {_html.escape(overall)}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    with st.expander("Create new customer"):
        new_name = st.text_input("Name", key="new_customer_name")
        new_slug = st.text_input("Slug", key="new_customer_slug", placeholder="e.g. acme-corp")
        if st.button("Create customer", use_container_width=True):
            if new_name and new_slug:
                with st.spinner("Creating..."):
                    result = api_post("/customers", json_body={"name": new_name, "slug": new_slug})
                if "error" in (result or {}):
                    st.error(result["error"])
                else:
                    st.success(f"Created: {new_name}")
                    st.session_state.customers = []
                    st.rerun()
            else:
                st.warning("Name and slug are required.")


# ── Guard: require a selected customer ───────────────────────────────────────

if not st.session_state.selected_customer:
    st.info("Select or create a customer in the sidebar to get started.")
    st.stop()

customer = st.session_state.selected_customer
customer_id = customer["slug"]   # backend expects slug everywhere, not numeric id
customer_name = customer["name"]

# ── Main header ───────────────────────────────────────────────────────────────

st.markdown(
    f'<div style="padding:8px 0 16px 0;border-bottom:1px solid #2a2a3a;margin-bottom:16px;">'
    f'<div style="font-size:18px;font-weight:600;color:#e8e8f0;">FDE Assistant</div>'
    f'<div style="font-size:12px;color:#55556a;margin-top:2px;">{_html.escape(customer_name)}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_brief, tab_exec, tab_query, tab_upload = st.tabs(
    ["Pre-Meeting Brief", "Exec 1:1 Brief", "Query", "Upload Documents"]
)


# ── Tab 1: Pre-Meeting Brief ──────────────────────────────────────────────────

with tab_brief:
    col_date, col_btn = st.columns([2, 1])
    with col_date:
        as_of = st.date_input(
            "As-of date (optional)",
            value=None,
            help="Leave blank to use today's date. Set a past date to generate a brief as of that date.",
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        generate_brief = st.button(
            f"Generate Brief for {customer_name}", use_container_width=True
        )

    if generate_brief:
        payload = {"customer_id": str(customer_id)}
        if as_of:
            payload["as_of_date"] = str(as_of)
        with st.spinner("Generating brief..."):
            result = api_post("/brief/pre-meeting", json_body=payload)
        if "error" in (result or {}):
            st.error(result["error"])
        else:
            st.session_state.pre_brief = result

    brief = st.session_state.pre_brief
    if brief and "error" not in brief:
        section_status = brief.get("section_status") or {}

        # ── Corpus warning (stale / empty corpus) ─────────────────────────────
        corpus_warning = brief.get("corpus_warning")
        if corpus_warning:
            st.markdown(
                f'<div style="padding:10px 14px;background:rgba(245,158,11,0.08);'
                f'border-left:3px solid #f59e0b;border-radius:6px;margin-bottom:12px;">'
                f'<span style="color:#f59e0b;font-size:13px;">⚠️ {_html.escape(corpus_warning)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Call snapshot bar ─────────────────────────────────────────────────
        overdue = brief.get("overdue_commitments") or []
        open_items = brief.get("open_items") or []
        posture = brief.get("recommended_posture") or []

        overdue_count = len(overdue)
        p0_count = sum(
            1 for item in open_items
            if (item.get("priority") or "").upper() in ("P0", "CRITICAL", "BLOCKER")
        )
        lead_item = next((d for d in posture if d.get("verb") == "Lead"), None)
        lead_directive = (lead_item or {}).get("directive", "") or ""
        lead_text = (lead_directive[:72] + "…") if len(lead_directive) > 72 else lead_directive

        as_of_date = brief.get("as_of_date", "")
        last_call_date = brief.get("last_call_date", "")

        overdue_color = "#ef4444" if overdue_count else "#22c55e"
        p0_color = "#ef4444" if p0_count else "#22c55e"

        snapshot_items_html = "".join(
            f'<div class="snapshot-item">'
            f'<div class="snapshot-number" style="color:{color};">{num}</div>'
            f'<div class="snapshot-label">{label}</div>'
            f'</div>'
            f'<div class="snapshot-divider"></div>'
            for num, label, color in [
                (str(overdue_count), "overdue", overdue_color),
                (str(p0_count), "P0 tickets", p0_color),
            ]
        )
        meta_html = '<div style="margin-left:auto;text-align:right;line-height:1.8;">'
        if as_of_date:
            meta_html += f'<div style="font-size:11px;color:#55556a;">As of {_html.escape(as_of_date)}</div>'
        if last_call_date:
            meta_html += f'<div style="font-size:11px;color:#55556a;">Last call: {_html.escape(last_call_date)}</div>'
        if lead_text:
            meta_html += (
                f'<div style="font-size:11px;color:#22c55e;margin-top:4px;">'
                f'🟢 Lead with: {_html.escape(lead_text)}</div>'
            )
        meta_html += '</div>'

        st.markdown(
            f'<div class="call-snapshot">{snapshot_items_html}{meta_html}</div>',
            unsafe_allow_html=True,
        )

        # ── 1. Overdue Commitments (always visible) ───────────────────────────
        if overdue:
            st.markdown(
                '<div style="padding:6px 12px;background:#3a1010;'
                'border-left:3px solid #ef4444;border-radius:6px;margin:8px 0 4px 0;">'
                '<strong style="color:#ef4444;font-size:13px;">⚠️ Overdue Commitments</strong>'
                '</div>',
                unsafe_allow_html=True,
            )
            for c in overdue:
                with st.container():
                    st.markdown('<div class="commitment-card">', unsafe_allow_html=True)
                    render_commitment(c)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── 2. Open Items (always visible, P0 first) ──────────────────────────
        st.markdown("#### Open Items")
        _priority_order = {"P0": 0, "CRITICAL": 0, "BLOCKER": 0, "P1": 1, "P2": 2}
        open_items_sorted = sorted(
            open_items,
            key=lambda x: _priority_order.get((x.get("priority") or "").upper(), 3),
        )
        status = section_status.get("open_items", "")
        if not open_items_sorted:
            st.markdown(section_status_label(status or "empty"), unsafe_allow_html=True)
        else:
            for item in open_items_sorted:
                with st.container():
                    st.markdown('<div class="open-item-card">', unsafe_allow_html=True)
                    render_open_item(item)
                    st.markdown('</div>', unsafe_allow_html=True)

        # ── 3. Recommended Posture (always visible, with grounding) ──────────
        st.markdown("#### Recommended Posture")
        status = section_status.get("recommended_posture", "")
        if not posture:
            st.markdown(section_status_label(status or "empty"), unsafe_allow_html=True)
        else:
            _verb_colors = {
                "Lead": "#27ae60",
                "Acknowledge": "#f59e0b",
                "Defer": "#3498db",
                "Push": "#e74c3c",
            }
            for d in posture:
                verb = d.get("verb", "")
                color = _verb_colors.get(verb, "#8888aa")
                grounding = d.get("grounding_item") or ""
                grounding_html = (
                    f'<div class="grounding-item">↳ {_html.escape(grounding)}</div>'
                    if grounding else ""
                )
                basis = d.get("basis") or ""
                basis_html = (
                    f'<div class="source-detail">{_html.escape(basis)}</div>'
                    if basis else ""
                )
                st.markdown(
                    f'<div style="padding:10px 14px;background:#16161f;border-radius:6px;'
                    f'border-left:3px solid {color};margin:6px 0;">'
                    f'<span style="color:{color};font-weight:700;font-size:11px;'
                    f'text-transform:uppercase;">{_html.escape(verb)}</span>'
                    f'<span style="color:#e8e8f0;font-size:13px;margin-left:8px;">'
                    f'{_html.escape(d.get("directive",""))}</span>'
                    f'{basis_html}'
                    f'{grounding_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── 4. Recent Changes (collapsed) ─────────────────────────────────────
        recent_changes = brief.get("recent_changes") or []
        rc_label = (
            f"Recent Changes ({len(recent_changes)})"
            if recent_changes else "Recent Changes — none"
        )
        with st.expander(rc_label):
            status = section_status.get("recent_changes", "")
            if not recent_changes:
                st.markdown(section_status_label(status or "empty"), unsafe_allow_html=True)
            else:
                for change in recent_changes:
                    what = change.get("what", "")
                    date_str = change.get("date", "")
                    customer_aware = change.get("customer_aware", False)
                    src = change.get("source") or {}
                    src_doc = src.get("document", "") if isinstance(src, dict) else ""

                    aware_tag = (
                        '<span style="font-size:10px;padding:1px 6px;background:#14331f;'
                        'color:#6ee7b7;border-radius:8px;margin-left:6px;">customer-visible</span>'
                        if customer_aware else ""
                    )
                    date_tag = (
                        f'<span style="color:#8888aa;font-size:11px;margin-left:8px;">'
                        f'{_html.escape(date_str)}</span>'
                        if date_str else ""
                    )
                    src_tag = (
                        f'<span class="source-detail"> — {_html.escape(src_doc)}</span>'
                        if src_doc else ""
                    )
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3a;">'
                        f'<span style="color:#e8e8f0;font-size:13px;">{_html.escape(what)}</span>'
                        f'{date_tag}{aware_tag}{src_tag}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── 5. Outstanding Commitments (collapsed) ────────────────────────────
        commitments = brief.get("outstanding_commitments") or []
        oc_label = (
            f"Outstanding Commitments ({len(commitments)})"
            if commitments else "Outstanding Commitments — none"
        )
        with st.expander(oc_label):
            status = section_status.get("outstanding_commitments", "")
            if not commitments:
                st.markdown(section_status_label(status or "empty"), unsafe_allow_html=True)
            else:
                rows = [
                    {
                        "Description": c.get("description", ""),
                        "Target Date": c.get("target_date") or c.get("promised_date") or "",
                        "Owner": c.get("owner", ""),
                        "Status": c.get("status", ""),
                        "Slipped": "Yes" if c.get("is_slipped") else "No",
                    }
                    for c in commitments
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # ── 6. Anticipated Questions (collapsed, with source_quote) ───────────
        questions = brief.get("anticipated_questions") or []
        aq_label = (
            f"Anticipated Questions ({len(questions)})"
            if questions else "Anticipated Questions — none"
        )
        with st.expander(aq_label):
            status = section_status.get("anticipated_questions", "")
            if not questions:
                st.markdown(section_status_label(status or "empty"), unsafe_allow_html=True)
            else:
                for q in questions:
                    urgency = q.get("urgency", "medium")
                    urgency_color = {
                        "high": "#ef4444", "medium": "#f59e0b", "low": "#6b7280"
                    }.get(urgency, "#6b7280")
                    topic = q.get("topic", "")
                    evidence = q.get("evidence", "")
                    source_quote = q.get("source_quote") or ""
                    evidence_html = (
                        f'<div class="source-detail">{_html.escape(evidence)}</div>'
                        if evidence else ""
                    )
                    quote_html = (
                        f'<div class="source-quote">"{_html.escape(source_quote)}"</div>'
                        if source_quote else ""
                    )
                    st.markdown(
                        f'<div style="padding:8px 0;border-bottom:1px solid #2a2a3a;">'
                        f'<span style="color:{urgency_color};font-size:10px;font-weight:600;'
                        f'text-transform:uppercase;margin-right:6px;">{_html.escape(urgency)}</span>'
                        f'<span style="color:#e8e8f0;font-size:13px;">{_html.escape(topic)}</span>'
                        f'{evidence_html}'
                        f'{quote_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── 7. Account Summary (collapsed) ────────────────────────────────────
        summary = brief.get("account_summary", "")
        with st.expander("Account Summary"):
            status = section_status.get("account_summary", "")
            if status in ("unavailable", "empty") and not summary:
                st.markdown(section_status_label(status), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="padding:12px 16px;background:#16161f;border-radius:8px;'
                    f'border-left:3px solid #6366f1;color:#c8c8e0;font-size:14px;'
                    f'line-height:1.65;">'
                    f'{_html.escape(summary)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── 8. Data Sources & Confidence (collapsed) ──────────────────────────
        stale_warnings = brief.get("stale_warnings") or []
        conflicts = brief.get("conflicts") or []
        section_sources = brief.get("section_sources") or {}
        section_data_as_of = brief.get("section_data_as_of") or {}

        ds_badges = []
        if stale_warnings:
            ds_badges.append(f"{len(stale_warnings)} stale")
        if conflicts:
            ds_badges.append(f"{len(conflicts)} conflict(s)")
        ds_label = "Data Sources & Confidence"
        if ds_badges:
            ds_label += f"  ⚠️ {', '.join(ds_badges)}"

        with st.expander(ds_label):
            if stale_warnings:
                st.markdown("**Stale source warnings**")
                for w in stale_warnings:
                    st.markdown(
                        f'<div class="confidence-warning">⚠️ {_html.escape(str(w))}</div>',
                        unsafe_allow_html=True,
                    )

            if section_sources:
                st.markdown("**Source provenance**")
                prov_rows = []
                for section, files in section_sources.items():
                    if not files:
                        continue
                    file_list = ", ".join(files[:3]) + ("…" if len(files) > 3 else "")
                    as_of_val = section_data_as_of.get(section, "")
                    prov_rows.append({
                        "Section": section.replace("_", " ").title(),
                        "Documents": file_list,
                        "As Of": as_of_val,
                    })
                if prov_rows:
                    st.dataframe(prov_rows, use_container_width=True, hide_index=True)

            if conflicts:
                st.markdown("**Conflicting claims**")
                for conflict in conflicts:
                    sa = conflict.get("source_a") or {}
                    sb = conflict.get("source_b") or {}
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3a;">'
                        f'<span style="color:#ef4444;font-size:11px;">⚡ conflict</span><br>'
                        f'<span style="color:#e8e8f0;font-size:12px;">'
                        f'"{_html.escape(conflict.get("claim_a",""))}"</span> '
                        f'<span style="color:#55556a;font-size:11px;">'
                        f'({_html.escape(sa.get("document",""))})</span><br>'
                        f'<span style="color:#8888aa;font-size:12px;">vs. '
                        f'"{_html.escape(conflict.get("claim_b",""))}"</span> '
                        f'<span style="color:#55556a;font-size:11px;">'
                        f'({_html.escape(sb.get("document",""))})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if not stale_warnings and not section_sources and not conflicts:
                st.caption("No confidence issues detected.")

    elif brief and "error" in brief:
        st.error(brief["error"])
    else:
        st.markdown(
            '<div class="not-found">'
            '<div style="font-size:14px;color:#8888aa;margin-bottom:6px;">'
            'Click "Generate Brief" to get started</div>'
            '<div style="font-size:12px;">Upload documents first to get the best results.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Tab 2: Exec 1:1 Brief ─────────────────────────────────────────────────────

with tab_exec:
    st.markdown(
        "Generate a brief for a 1:1 with a specific executive or stakeholder at this account."
    )

    col_person, col_btn2 = st.columns([2, 1])
    with col_person:
        people_result = api_get(f"/customers/{customer_id}/people")
        people = []
        if isinstance(people_result, list):
            people = people_result
        elif isinstance(people_result, dict) and "error" not in people_result:
            people = people_result.get("people") or []

        if people:
            person_names = [p.get("name", str(p.get("id", ""))) for p in people]
            selected_person_name = st.selectbox("Select person", options=person_names)
            selected_person = next(
                (p for p in people if p.get("name") == selected_person_name), None
            )
            person_id = selected_person.get("id") if selected_person else None
        else:
            st.caption("No people found for this customer. Add one below.")
            person_id = None

    with col_btn2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        generate_exec = st.button("Generate 1:1 Brief", use_container_width=True)

    with st.expander("Add stakeholder"):
        new_name = st.text_input("Name", key="new_person_name")
        new_role = st.text_input("Role (optional)", key="new_person_role")
        new_email = st.text_input("Email (optional)", key="new_person_email")
        if st.button("Add", key="add_person_btn"):
            if not new_name.strip():
                st.warning("Name is required.")
            else:
                add_result = api_post(
                    f"/customers/{customer_id}/people",
                    json_body={"name": new_name.strip(), "role": new_role.strip() or None,
                               "email": new_email.strip() or None},
                )
                if add_result and "error" not in add_result:
                    st.success(f"Added {new_name.strip()}. Reload to select them.")
                    st.rerun()
                else:
                    st.error((add_result or {}).get("error", "Failed to add person."))

    if generate_exec:
        if not person_id:
            st.warning("Select a person above, or add one first.")
        else:
            payload = {"customer_id": str(customer_id), "person_id": str(person_id)}
            with st.spinner("Generating exec brief..."):
                result = api_post("/brief/exec-1on1", json_body=payload)
            if "error" in (result or {}):
                st.error(result["error"])
            else:
                st.session_state.exec_brief = result

    exec_brief = st.session_state.exec_brief
    if exec_brief and "error" not in exec_brief:
        stale_warnings = exec_brief.get("stale_warnings") or []
        if stale_warnings:
            for w in stale_warnings:
                st.markdown(
                    f'<div class="confidence-warning">⚠️ {_html.escape(str(w))}</div>',
                    unsafe_allow_html=True,
                )

        # ── 1. Role & Tenure ───────────────────────────────────────────────────
        st.markdown("#### Role & Tenure")
        role_text = exec_brief.get("role_and_tenure", "")
        if role_text:
            st.markdown(
                f'<div style="padding:10px 14px;background:#16161f;border-left:3px solid #6366f1;'
                f'border-radius:6px;color:#c8c8e0;font-size:14px;line-height:1.6;">'
                f'{_html.escape(role_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#888;font-size:12px;">No role information available</span>',
                unsafe_allow_html=True,
            )

        # ── 2. Stated Position ─────────────────────────────────────────────────
        st.markdown("#### Stated Position")
        statements = exec_brief.get("stated_position") or []
        if statements:
            _sentiment_colors = {
                "positive": "#22c55e",
                "neutral": "#8888aa",
                "concern": "#f59e0b",
                "request": "#3498db",
            }
            for stmt in statements:
                content = stmt.get("content", "")
                said_by = stmt.get("said_by", "other")
                src = stmt.get("source") or {}
                src_doc = src.get("document", "") if isinstance(src, dict) else ""
                stated_date = stmt.get("stated_date") or ""
                sentiment = stmt.get("sentiment") or ""
                verification = stmt.get("verification") or {}
                flag = verification.get("flag") if isinstance(verification, dict) else None
                flag_tag = {
                    "stale_source": " ⚠️ stale",
                    "verify_before_quoting": " 🔍 verify",
                    "conflict": " ⚡ conflict",
                }.get(flag, "")

                sentiment_html = ""
                if sentiment:
                    sc = _sentiment_colors.get(sentiment, "#8888aa")
                    sentiment_html = (
                        f'<span style="font-size:10px;padding:1px 6px;'
                        f'background:{sc}22;color:{sc};border-radius:8px;'
                        f'margin-right:6px;">{_html.escape(sentiment)}</span>'
                    )

                meta_parts = [p for p in [stated_date, src_doc] if p]
                meta_html = (
                    f'<div class="source-detail">{_html.escape(" · ".join(meta_parts))}</div>'
                    if meta_parts else ""
                )

                if said_by == "person":
                    st.markdown(
                        f'<blockquote style="border-left:3px solid #6366f1;padding:6px 12px;'
                        f'color:#c8c8e0;margin:8px 0;">'
                        f'{sentiment_html}"{_html.escape(content)}"{flag_tag}'
                        f'{meta_html}'
                        f'</blockquote>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3a;">'
                        f'{sentiment_html}'
                        f'<span style="color:#e8e8f0;font-size:13px;">'
                        f'{_html.escape(content)}</span>{flag_tag}'
                        f'{meta_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<span style="color:#888;font-size:12px;">No statements on record</span>',
                unsafe_allow_html=True,
            )

        # ── 3. Recent Signals ──────────────────────────────────────────────────
        st.markdown("#### Recent Signals")
        signals = exec_brief.get("recent_signals") or []
        if signals:
            for sig in signals:
                event = sig.get("event", "")
                date_str = sig.get("date", "")
                src = sig.get("source") or {}
                src_doc = src.get("document", "") if isinstance(src, dict) else ""
                date_tag = (
                    f'<span style="color:#8888aa;font-size:11px;margin-left:8px;">'
                    f'{_html.escape(date_str)}</span>'
                ) if date_str else ""
                src_tag = (
                    f'<span class="source-detail"> — {_html.escape(src_doc)}</span>'
                ) if src_doc else ""
                st.markdown(
                    f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3a;">'
                    f'<span style="color:#e8e8f0;font-size:13px;">{_html.escape(event)}</span>'
                    f'{date_tag}{src_tag}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="color:#888;font-size:12px;">No recent signals</span>',
                unsafe_allow_html=True,
            )

        # ── 4. Open Asks ───────────────────────────────────────────────────────
        st.markdown("#### Open Asks")
        asks = exec_brief.get("open_asks") or []
        if asks:
            rows = [
                {
                    "Ask": a.get("ask", ""),
                    "Date": a.get("date", ""),
                    "Status": a.get("status", "open"),
                }
                for a in asks
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<span style="color:#888;font-size:12px;">No open asks on record</span>',
                unsafe_allow_html=True,
            )

        # ── 5. Recommended Approach ────────────────────────────────────────────
        st.markdown("#### Recommended Approach")
        approach = exec_brief.get("recommended_approach", "")
        if approach:
            st.markdown(
                f'<div style="padding:12px 16px;background:#16161f;border-left:3px solid #27ae60;'
                f'border-radius:6px;color:#c8c8e0;font-size:14px;line-height:1.65;">'
                f'{_html.escape(approach)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#888;font-size:12px;">No recommendation available</span>',
                unsafe_allow_html=True,
            )

        conflicts = exec_brief.get("conflicts") or []
        if conflicts:
            with st.expander(f"Conflicting information ({len(conflicts)})"):
                for conflict in conflicts:
                    st.markdown(
                        f"- **{conflict.get('claim_a','')}** vs **{conflict.get('claim_b','')}**"
                    )

    elif exec_brief and "error" in exec_brief:
        st.error(exec_brief["error"])
    else:
        st.markdown(
            '<div class="not-found">'
            'Select a person and click "Generate 1:1 Brief"'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Tab 3: Query ──────────────────────────────────────────────────────────────

with tab_query:
    st.markdown(
        "Ask a focused question about this customer. The system searches uploaded documents "
        "and returns an answer with source citations."
    )

    question = st.text_input(
        "Question",
        placeholder="e.g. What did we agree on in the last call? What's the current SLA?",
        label_visibility="collapsed",
    )

    if st.button("Submit", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching..."):
                result = api_post(
                    "/query",
                    json_body={"question": question, "customer_id": str(customer_id)},
                )
            if "error" in (result or {}):
                st.error(result["error"])
            else:
                st.session_state.query_result = result

    qr = st.session_state.query_result
    if qr and "error" not in qr:
        answer_status = qr.get("answer_status", "not_found")
        answer = qr.get("answer", "")
        answer_as_of = qr.get("answer_as_of") or ""
        confidence_explanation = qr.get("confidence_explanation") or ""
        sources_searched = qr.get("sources_searched", 0)
        all_citations = qr.get("citations") or []
        primary_citation = qr.get("citation")

        # ── Status badge + as-of date ─────────────────────────────────────────
        status_colors = {
            "ok": "#22c55e", "partial": "#f59e0b",
            "not_found": "#ef4444", "error": "#ef4444",
        }
        status_color = status_colors.get(answer_status, "#6b7280")
        as_of_html = (
            f' <span style="font-size:11px;color:#55556a;margin-left:8px;">'
            f'as of {_html.escape(answer_as_of)}</span>'
            if answer_as_of else ""
        )
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<span style="display:inline-block;padding:3px 10px;'
            f'background:{status_color}22;color:{status_color};'
            f'border:1px solid {status_color};border-radius:10px;'
            f'font-size:11px;font-weight:600;letter-spacing:0.3px;">'
            f'{answer_status.upper()}</span>'
            f'{as_of_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Confidence explanation (amber strip for partial/error) ────────────
        if confidence_explanation:
            st.markdown(
                f'<div class="confidence-warning">⚠️ {_html.escape(confidence_explanation)}</div>',
                unsafe_allow_html=True,
            )

        # ── Answer block ──────────────────────────────────────────────────────
        if answer:
            st.markdown(
                f'<div style="padding:14px 16px;background:#16161f;border:1px solid #2a2a3a;'
                f'border-left:3px solid #4f6ef7;border-radius:6px;margin:8px 0 12px 0;'
                f'color:#ffffff;font-size:14px;line-height:1.6;">'
                f'{_html.escape(answer)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="not-found">No answer found in the uploaded documents.</div>',
                unsafe_allow_html=True,
            )

        # ── Missing doc type hint ────────────────────────────────────────────
        missing = qr.get("missing_doc_types") or []
        if missing:
            st.markdown(
                f'<div style="font-size:12px;color:#8888aa;margin-bottom:8px;">'
                f'💡 Uploading a <strong style="color:#c8c8e0;">{_html.escape(missing[0])}</strong>'
                f' could improve results for this question.</div>',
                unsafe_allow_html=True,
            )

        # ── Citations (collapsed) ─────────────────────────────────────────────
        cit_count = len(all_citations) or (1 if primary_citation else 0)
        cit_label = f"Sources ({cit_count})"
        if sources_searched:
            cit_label += f" — {sources_searched} chunks searched"
        with st.expander(cit_label):
            display_citations = all_citations or (
                [primary_citation] if primary_citation and isinstance(primary_citation, dict) else []
            )
            if display_citations:
                for c in display_citations:
                    doc = c.get("document", "")
                    doc_date = c.get("doc_date", "")
                    is_stale = c.get("is_stale", False)
                    stale_tag = " ⚠️ stale" if is_stale else ""
                    location = c.get("location", "")
                    loc_html = (
                        f'<span class="source-detail"> · {_html.escape(location)}</span>'
                        if location else ""
                    )
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3a;">'
                        f'<span style="color:#e8e8f0;font-size:13px;">{_html.escape(doc)}</span>'
                        f'{loc_html}'
                        f'<div class="source-detail">{_html.escape(doc_date)}'
                        f'<span style="color:#f59e0b;">{_html.escape(stale_tag)}</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No source citations available.")

        # ── Conflicts (collapsed, only if present) ────────────────────────────
        conflicts = qr.get("conflicts") or []
        if conflicts:
            with st.expander(f"Conflicting information ({len(conflicts)})"):
                for conflict in conflicts:
                    sa = conflict.get("source_a") or {}
                    sb = conflict.get("source_b") or {}
                    st.markdown(
                        f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3a;">'
                        f'<span style="color:#ef4444;font-size:11px;">⚡ conflict</span><br>'
                        f'<span style="color:#e8e8f0;font-size:12px;">'
                        f'"{_html.escape(conflict.get("claim_a",""))}"</span> '
                        f'<span style="color:#55556a;font-size:11px;">'
                        f'({_html.escape(sa.get("document",""))})</span><br>'
                        f'<span style="color:#8888aa;font-size:12px;">vs. '
                        f'"{_html.escape(conflict.get("claim_b",""))}"</span> '
                        f'<span style="color:#55556a;font-size:11px;">'
                        f'({_html.escape(sb.get("document",""))})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    elif qr and "error" in qr:
        st.error(qr["error"])
    else:
        st.markdown(
            '<div class="not-found">'
            "Enter a question above to search this customer's documents"
            '</div>',
            unsafe_allow_html=True,
        )


# ── Tab 4: Upload Documents ───────────────────────────────────────────────────

with tab_upload:
    st.markdown(f"Upload documents for **{customer_name}**.")

    st.markdown(
        '<div style="padding:10px 14px;background:#16161f;border:1px solid #2a2a3a;'
        'border-left:3px solid #f59e0b;border-radius:6px;margin-bottom:16px;">'
        '<strong style="color:#f59e0b;">Required filename format:</strong><br>'
        '<code style="color:#e8e8f0;font-size:13px;">YYYY-MM-DD_doctype_descriptor.ext</code><br>'
        '<span style="color:#8888aa;font-size:11px;">'
        'Example: 2024-03-15_transcript_qbr-call.pdf</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _DOC_TYPES = [
        "transcript",
        "ticket",
        "qbr_deck",
        "commitment_tracker",
        "account_notes",
        "solution_architecture",
    ]

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "html", "txt", "json", "csv"],
        help=(
            "Supported: .pdf, .docx, .html, .txt, .json, .csv. "
            "For best results, name files like YYYY-MM-DD_doctype_descriptor.ext"
        ),
    )

    doc_type = st.selectbox(
        "Document type",
        options=_DOC_TYPES,
        help="Select the type of document being uploaded.",
    )

    replace_existing = st.checkbox(
        "Replace if already uploaded",
        value=False,
        help="If a document with the same filename exists, delete it and re-index.",
    )

    if uploaded_file and st.button("Upload", use_container_width=True):
        with st.spinner("Uploading and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {
                "doc_type": doc_type,
                "replace": "true" if replace_existing else "false",
            }
            result = api_post_files(f"/customers/{customer_id}/upload", files=files, data=data)

        if "error" in (result or {}):
            st.error(result["error"])
        else:
            chunks = result.get("chunks", "?")
            doc_date = result.get("doc_date", "")
            msg = f"Uploaded successfully. Chunks indexed: {chunks}."
            if doc_date:
                msg += f" Document date: {doc_date}."
            st.success(msg)
            st.session_state.corpus_health = None

    if not uploaded_file:
        st.markdown(
            '<div style="padding:32px;text-align:center;border:1px dashed #2a2a3a;'
            'border-radius:8px;color:#55556a;font-size:13px;margin-top:16px;">'
            'Select a file above to upload</div>',
            unsafe_allow_html=True,
        )
