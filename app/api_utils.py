import requests
import streamlit as st
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def get_api_response(question, session_id, model,
                     notify_slack=False, notify_email=None,
                     user_id="default"):             # ← add user_id param
    data = {
        "question":     question,
        "model":        model,
        "notify_slack": notify_slack,
        "notify_email": notify_email,
        "user_id":      user_id                      # ← include in payload
    }
    if session_id:
        data["session_id"] = session_id
    try:
        response = requests.post(f"{API_BASE_URL}/chat",
                                 headers={"Content-Type": "application/json"},
                                 json=data)
        if response.status_code == 200:
            return response.json()
        st.error(f"API error {response.status_code}: {response.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def upload_document(file, user_id="default"):        # ← add user_id param
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload-doc",
                                  files=files,
                                  params={"user_id": user_id})  # ← query param
        if response.status_code == 200:
            return response.json()
        st.error(f"Upload failed: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def list_documents(user_id="default"):               # ← add user_id param
    try:
        response = requests.get(f"{API_BASE_URL}/list-docs",
                                 params={"user_id": user_id})
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

def delete_document(file_id, user_id="default"):     # ← add user_id param
    try:
        response = requests.post(f"{API_BASE_URL}/delete-doc",
                                 headers={"Content-Type": "application/json"},
                                 json={"file_id": file_id, "user_id": user_id})
        if response.status_code == 200:
            return response.json()
        st.error(f"Delete failed: {response.status_code}")
        return None
    except Exception as e:
        st.error(f"Delete error: {str(e)}")
        return None

def get_analytics(user_id="default"):
    try:
        response = requests.get(f"{API_BASE_URL}/analytics",
                                 params={"user_id": user_id})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")
        return None

def get_audit_log(limit=100, user_id="default"):
    try:
        response = requests.get(f"{API_BASE_URL}/audit-log",
                                 params={"limit": limit, "user_id": user_id})
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

def answer_questionnaire(file, user_id="default"):
    try:
        files = {"file": (file.name, file, "text/csv")}
        response = requests.post(f"{API_BASE_URL}/answer-questionnaire",
                                  files=files,
                                  params={"user_id": user_id})
        if response.status_code == 200:
            return response.json()
        st.error(f"Questionnaire error: {response.status_code}")
        return None
    except Exception as e:
        st.error(f"Questionnaire error: {str(e)}")
        return None