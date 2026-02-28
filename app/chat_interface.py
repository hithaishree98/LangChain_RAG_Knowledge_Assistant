import streamlit as st
from api_utils import get_api_response

def display_chat_interface():
    with st.expander("📤 Send answer to Slack", expanded=False):
        notify_slack = st.checkbox("Send to Slack channel", key="notify_slack")
        if notify_slack:
            # Check health endpoint to see if Slack is configured on backend
            try:
                import requests, os
                API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
                health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
                slack_status = health.get("checks", {}).get("slack", "unknown")
                if slack_status == "configured":
                    st.success("✅ Slack is configured — answers will be posted to your channel")
                else:
                    st.warning("⚠️ Slack webhook not configured on backend. Add SLACK_WEBHOOK_URL to your .env")
            except Exception:
                st.caption("📤 Answer will be sent to Slack if configured on backend.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "meta" in message:
                meta = message["meta"]
                conf = meta.get("confidence", 0)
                color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                col1, col2 = st.columns(2)
                col1.caption(f"{color} Confidence: {int(conf * 100)}%")
                sources = meta.get("sources", [])
                if sources:
                    col2.caption(f"📄 Source: {', '.join(sources)}")
                if meta.get("escalated"):
                    st.warning("⚠️ Low confidence — consider uploading more relevant documents.")


    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Searching documents..."):
            response = get_api_response(
                prompt,
                st.session_state.session_id,
                st.session_state.model,
                notify_slack=st.session_state.get("notify_slack", False)
            )
            if response:
                st.session_state.session_id = response.get("session_id")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "meta": {
                        "confidence": response.get("confidence", 0),
                        "sources": response.get("sources", []),
                        "escalated": response.get("escalated", False)
                    }
                })
                st.rerun()
            else:
                st.error("Failed to get response. Is the API running?")