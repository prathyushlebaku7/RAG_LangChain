import streamlit as st
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from embeddings import clear_faiss_index, save_uploaded_files, embed_files_from_paths, load_faiss_index
from generation import get_llm_chain, speak_text, listen_query

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ask Your PDFs", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠 Talk to Your PDFs (FAISS Version)</h1>", unsafe_allow_html=True)

# --- CUSTOM STYLES ---
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stExpanderHeader {
            font-size: 18px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "embedding_created" not in st.session_state:
    st.session_state.embedding_created = False

# --- SIDEBAR STATUS ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.markdown(f"**✅ Embeddings Created:** {st.session_state.embedding_created}")
    st.markdown(f"**💬 Chat Turns:** `{len(st.session_state.chat_history)}`")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📁 Upload PDFs", "🎙️ Ask Question", "📜 Chat History"])

# --- UPLOAD PDFs TAB ---
with tab1:
    st.subheader("📤 Upload PDFs (clears old FAISS index)")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and not st.session_state.embedding_created:
        clear_faiss_index()
        file_paths = save_uploaded_files(uploaded_files)
        success = embed_files_from_paths(file_paths)
        if success:
            st.success("✅ PDFs embedded and saved using FAISS!")
            st.session_state.embedding_created = True

# --- LOAD FAISS RETRIEVER ---
try:
    retriever = load_faiss_index()
except Exception as e:
    st.error(f"❌ Could not load FAISS index: {e}")
    st.stop()

qa_chain = get_llm_chain(retriever, st.session_state.memory)

# --- ASK QUESTION TAB ---
with tab2:
    st.subheader("🎤 Press button and ask your question")
    query = None
    if st.button("🎙️ Press button and ask your question"):
        st.info("🎧 Voice interaction started. Say 'goodbye' to end.")
        query = None

        while True:
            query= listen_query(timeout=6)

            if query.lower().strip() == "goodbye":
                st.success("👋 Ending session as requested.")
                break

            if query and not query.startswith("ERROR::"):
                st.success(f"✅ You said: `{query}`")
                with st.spinner("🔍 Searching the brain..."):
                    try:
                        result = qa_chain.invoke({"question": query})
                        answer = result["answer"]
                        sources = result.get("source_documents", [])

                        st.session_state.chat_history.append((query, answer, sources))
                        speak_text(answer)

                        with open("chat_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"\n\n[{datetime.now()}]\nQ: {query}\nA: {answer}\n")

                        st.success("✅ Answer generated!")

                    except Exception as e:
                        st.error(f"💥 Error during answer generation: {e}")
            else:
                st.warning("⚠️ Could not capture your query.")


# --- CHAT HISTORY TAB ---
with tab3:
    st.subheader("📜 Previous Conversations")
    if st.session_state.chat_history:
        for i, (q, a, sources) in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"❓ Q{i}: {q}"):
                st.markdown(f"**🧠 Answer:** {a}")
                if sources:
                    st.markdown("**📚 Sources:**")
                    for doc in sources:
                        st.markdown(f"• `{doc.metadata.get('source', 'Unknown')}`")
                        st.code(doc.page_content[:300] + "...", language="markdown")
    else:
        st.info("💡 No chat history yet. Start by uploading a PDF and asking a question.")


