import streamlit as st
from api_utils import upload_document, list_documents, delete_document

def display_sidebar():
    model_options = ["llama-3.1-8b-instant"]

    
    selected = st.sidebar.selectbox("Select Model", options=model_options)
    st.session_state.model = selected

    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a file",
                                              type=["pdf", "docx", "html"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(
                        f"'{uploaded_file.name}' uploaded. ID: {upload_response['file_id']}"
                    )
                    st.session_state.documents = list_documents()

    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()

    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']})")
        selected_file_id = st.sidebar.selectbox(
            "Select document to delete",
            options=[doc['id'] for doc in documents],
            format_func=lambda x: next(
                doc['filename'] for doc in documents if doc['id'] == x
            )
        )
        if st.sidebar.button("Delete Selected Document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Deleted document ID {selected_file_id}")
                    st.session_state.documents = list_documents()
                else:
                    st.sidebar.error(f"Failed to delete ID {selected_file_id}")