import os
import streamlit as st

from document_loader import extract_text_from_pdf

from vector_store import (
    create_chunks,
    create_vector_store,
    delete_vector_store
)

from rag_pipeline import answer_question


# ==================================================
# SETTINGS
# ==================================================

MAX_FILE_SIZE_MB = 10

MAX_FILE_SIZE_BYTES = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "documents_processed" not in st.session_state:

    st.session_state.documents_processed = False


if "processed_files" not in st.session_state:

    st.session_state.processed_files = []


if "uploader_key" not in st.session_state:

    st.session_state.uploader_key = 0


# ==================================================
# TITLE
# ==================================================

st.title(
    "📚 Domain-Specific RAG Chatbot"
)

st.write(
    "Upload one or more PDF documents and ask "
    "questions based only on their content."
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    st.caption(
        f"Maximum file size: {MAX_FILE_SIZE_MB} MB per PDF"
    )

    # --------------------------------------------------
    # SHOW UPLOADED FILES
    # --------------------------------------------------

    if uploaded_files:

        st.subheader(
            "Selected Documents"
        )

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

    # --------------------------------------------------
    # PROCESS BUTTON
    # --------------------------------------------------

    process_button = st.button(
        "🔄 Process Documents",
        use_container_width=True
    )

    # --------------------------------------------------
    # CLEAR DOCUMENTS
    # --------------------------------------------------

    clear_documents_button = st.button(
        "🗑️ Clear Documents",
        use_container_width=True
    )

    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    clear_chat_button = st.button(
        "💬 Clear Chat",
        use_container_width=True
    )


# ==================================================
# CLEAR CHAT
# ==================================================

if clear_chat_button:

    st.session_state.messages = []

    st.rerun()


# ==================================================
# CLEAR DOCUMENTS
# ==================================================

if clear_documents_button:

    delete_vector_store()

    st.session_state.documents_processed = False

    st.session_state.processed_files = []

    st.session_state.messages = []

    # Change uploader key to reset the uploader
    st.session_state.uploader_key += 1

    st.success(
        "Documents and chat history cleared."
    )

    st.rerun()


# ==================================================
# PROCESS DOCUMENTS
# ==================================================

if process_button:

    # --------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF file."
        )

    else:

        valid_files = []

        validation_errors = []

        # --------------------------------------------------
        # VALIDATE EACH FILE
        # --------------------------------------------------

        for file in uploaded_files:

            # Check extension
            if not file.name.lower().endswith(".pdf"):

                validation_errors.append(
                    f"{file.name}: Only PDF files are allowed."
                )

                continue

            # Check size
            if file.size > MAX_FILE_SIZE_BYTES:

                validation_errors.append(
                    f"{file.name}: File exceeds "
                    f"{MAX_FILE_SIZE_MB} MB."
                )

                continue

            valid_files.append(file)

        # --------------------------------------------------
        # SHOW VALIDATION ERRORS
        # --------------------------------------------------

        if validation_errors:

            for error in validation_errors:

                st.error(error)

        # --------------------------------------------------
        # PROCESS VALID FILES
        # --------------------------------------------------

        if valid_files:

            with st.spinner(
                "Extracting text and building the vector store..."
            ):

                try:

                    # Always replace the previous document collection
                    delete_vector_store()

                    all_documents = []

                    # ==========================================
                    # TEXT EXTRACTION
                    # ==========================================

                    for uploaded_file in valid_files:

                        documents = extract_text_from_pdf(
                            uploaded_file
                        )

                        if documents:

                            all_documents.extend(
                                documents
                            )

                        else:

                            st.warning(
                                f"No readable text found in "
                                f"{uploaded_file.name}."
                            )

                    # ==========================================
                    # CHECK EXTRACTION
                    # ==========================================

                    if not all_documents:

                        st.error(
                            "No readable text was found "
                            "in the uploaded PDF files."
                        )

                        st.session_state.documents_processed = False

                    else:

                        # ==========================================
                        # CHUNKING
                        # ==========================================

                        chunks = create_chunks(
                            all_documents
                        )

                        # ==========================================
                        # EMBEDDINGS + FAISS
                        # ==========================================

                        number_of_chunks = create_vector_store(
                            chunks
                        )

                        # ==========================================
                        # UPDATE STATE
                        # ==========================================

                        st.session_state.documents_processed = True

                        st.session_state.processed_files = [
                            file.name
                            for file in valid_files
                        ]

                        st.session_state.messages = []

                        st.success(
                            "Documents processed successfully!"
                        )

                        st.info(
                            f"Processed "
                            f"{len(valid_files)} PDF(s), "
                            f"extracted "
                            f"{len(all_documents)} page(s), "
                            f"and created "
                            f"{number_of_chunks} text chunks."
                        )

                except Exception as error:

                    st.error(
                        f"An error occurred while processing "
                        f"the documents:\n\n{error}"
                    )

                    st.session_state.documents_processed = False


# ==================================================
# DOCUMENT STATUS
# ==================================================

if st.session_state.documents_processed:

    st.success(
        "✅ Document collection is ready for questions."
    )

    with st.expander(
        "View processed documents"
    ):

        for filename in st.session_state.processed_files:

            st.write(
                f"📄 {filename}"
            )

else:

    st.info(
        "Upload PDF files in the sidebar and click "
        "'Process Documents' before asking questions."
    )


# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)

        # Show sources for assistant messages
        if role == "assistant":

            sources = message.get(
                "sources",
                []
            )

            if sources:

                st.markdown(
                    "**Sources:**"
                )

                for source in sources:

                    st.caption(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)


if question:

    # --------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------

    if not st.session_state.documents_processed:

        st.warning(
            "Please upload and process your documents first."
        )

    else:

        # ==============================================
        # USER MESSAGE
        # ==============================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ==============================================
        # ASSISTANT RESPONSE
        # ==============================================

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the uploaded documents..."
            ):

                try:

                    result = answer_question(
                        question
                    )

                    answer = result[
                        "answer"
                    ]

                    sources = result[
                        "sources"
                    ]

                    # ------------------------------
                    # ANSWER
                    # ------------------------------

                    st.markdown(
                        answer
                    )

                    # ------------------------------
                    # SOURCES
                    # ------------------------------

                    if sources:

                        st.markdown(
                            "**Sources:**"
                        )

                        for source in sources:

                            st.caption(
                                f"📄 "
                                f"{source['source']} "
                                f"— Page "
                                f"{source['page']}"
                            )

                    # ------------------------------
                    # SAVE CHAT
                    # ------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                except Exception as error:

                    error_message = (
                        "An error occurred while generating "
                        f"the answer: {error}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        }
                    )


# ==================================================
# RESPONSIBLE AI NOTICE
# ==================================================

st.divider()

st.caption(
    "⚠️ Responsible AI Notice: This chatbot answers "
    "questions using the uploaded documents, but "
    "generated answers may still contain errors. "
    "Verify important or high-stakes information "
    "against the original document."
)