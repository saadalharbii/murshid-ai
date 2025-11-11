"""
Streamlit frontend for MurshidAI.
Provides chat interface and admin panel for data management.
"""

import streamlit as st
import requests
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="MurshidAI - مرشد",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL support and better styling
st.markdown("""
<style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .ltr {
        direction: ltr;
        text-align: left;
    }
    /* User messages - dark blue with white text */
    .stChatMessage[data-testid="user-message"] {
        background-color: #0066cc !important;
        color: white !important;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* Assistant messages - light background with dark text */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* General chat message styling (fallback) */
    .stChatMessage {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .source-box {
        background-color: #e8f4f8;
        border-left: 4px solid #0066cc;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        color: #1f1f1f;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True


def query_chatbot(question: str) -> Dict[str, Any]:
    """
    Send query to the backend API.

    Args:
        question: User's question

    Returns:
        API response as dictionary
    """
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None


def upload_text(content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Upload text content via admin API.

    Args:
        content: Text content
        metadata: Optional metadata

    Returns:
        API response as dictionary
    """
    try:
        response = requests.post(
            f"{API_URL}/api/admin/upload/text",
            json={"content": content, "metadata": metadata or {}},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading text: {e}")
        return None


def upload_html_files(files) -> Dict[str, Any]:
    """
    Upload HTML files via admin API.

    Args:
        files: List of uploaded files

    Returns:
        API response as dictionary
    """
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), "text/html")))

        response = requests.post(
            f"{API_URL}/api/admin/upload/html",
            files=files_data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading HTML files: {e}")
        return None


def get_stats() -> Dict[str, Any]:
    """
    Get database statistics.

    Returns:
        Stats dictionary
    """
    try:
        response = requests.get(f"{API_URL}/api/admin/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting stats: {e}")
        return None


def display_sources(sources: List[Dict[str, Any]]):
    """Display retrieved sources."""
    if sources:
        with st.expander(f"📚 View Sources ({len(sources)})", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Source {i}</strong> (Similarity: {source.get('similarity_score', 0):.2%})
                    <br><br>
                    {source.get('content', '')[:300]}...
                    <br><br>
                    <small>📅 {source.get('metadata', {}).get('date_range', 'N/A')} |
                    👤 {source.get('metadata', {}).get('authors', 'Unknown')}</small>
                </div>
                """, unsafe_allow_html=True)


def chat_page():
    """Main chat interface page."""
    st.title("🎓 MurshidAI - مرشد")
    st.markdown("### مساعد ذكي للطلاب المبتعثين | AI Assistant for Scholarship Students")

    # Settings in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.show_sources = st.checkbox("Show sources", value=True)

        st.markdown("---")
        st.markdown("### About")
        st.info("""
        **MurshidAI** helps Saudi scholarship students in the UK by answering
        questions based on community knowledge from Telegram groups, FAQs, and notes.

        **Supported Languages:**
        - Arabic (العربية)
        - English
        """)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                if st.session_state.show_sources and message["sources"]:
                    display_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question in Arabic or English..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_chatbot(prompt)

                if response:
                    answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                    language = response.get("language", "unknown")
                    sources = response.get("sources", [])
                    query_time = response.get("query_time", 0)

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if st.session_state.show_sources and sources:
                        display_sources(sources)

                    # Display query time
                    st.caption(f"⏱️ Response time: {query_time:.2f}s | 🌐 Language: {language}")

                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = "Error: Could not get response from server."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def admin_page():
    """Admin panel for data management."""
    st.title("🔧 Admin Panel")

    # Stats section
    st.header("📊 Database Statistics")
    if st.button("Refresh Stats"):
        stats = get_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Status", stats.get("status", "unknown"))

    st.markdown("---")

    # Upload text section
    st.header("📝 Upload Text Content")
    st.markdown("Manually add FAQs, notes, or any text content.")

    text_content = st.text_area(
        "Content (Arabic or English)",
        height=150,
        placeholder="Enter your content here..."
    )

    col1, col2 = st.columns(2)
    with col1:
        metadata_source = st.text_input("Source", value="manual")
    with col2:
        metadata_type = st.text_input("Type", value="faq")

    if st.button("Upload Text"):
        if text_content:
            with st.spinner("Uploading..."):
                metadata = {
                    "source": metadata_source,
                    "type": metadata_type
                }
                result = upload_text(text_content, metadata)

                if result and result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                else:
                    st.error("❌ Upload failed")
        else:
            st.warning("Please enter content to upload")

    st.markdown("---")

    # Upload HTML files section
    st.header("📄 Upload Telegram HTML Files")
    st.markdown("Upload exported Telegram chat HTML files.")

    uploaded_files = st.file_uploader(
        "Choose HTML files",
        type=["html"],
        accept_multiple_files=True
    )

    if st.button("Upload HTML Files"):
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                result = upload_html_files(uploaded_files)

                if result and result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                    st.info(f"Documents processed: {result.get('documents_processed', 0)}")
                else:
                    st.error("❌ Upload failed")
        else:
            st.warning("Please select HTML files to upload")


def main():
    """Main application."""
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["💬 Chat", "🔧 Admin Panel"])

    if page == "💬 Chat":
        chat_page()
    elif page == "🔧 Admin Panel":
        admin_page()


if __name__ == "__main__":
    main()
