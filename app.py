# FILE: app.py
import streamlit as st
from datetime import datetime
from langchain.memory import ConversationBufferMemory
# Assuming embeddings.py and generation.py are in the same directory
from embeddings import clear_faiss_index, save_uploaded_files, embed_files_from_paths, load_faiss_index
from generation import get_llm_chain, speak_text, listen_query
import os # Added for checking FAISS index existence
# import time # Needed only if using time.sleep

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
            padding: 0.5rem 1rem; /* Added padding for better look */
        }
        .stExpanderHeader {
            font-size: 18px !important;
        }
        .stSpinner > div > div { /* Style spinner */
            border-top-color: #4CAF50;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' # Explicitly define output key for clarity
    )
if "embedding_created" not in st.session_state:
    # Check if FAISS index exists from a previous run before setting to False
    st.session_state.embedding_created = os.path.exists("./faiss_index")
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- SIDEBAR STATUS ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.markdown(f"**✅ Embeddings Ready:** {'Yes' if st.session_state.embedding_created else 'No'}")
    st.markdown(f"**💬 Chat Turns:** `{len(st.session_state.chat_history)}`")
    if st.button("Clear Chat History & Memory"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun() # Rerun to reflect the cleared state immediately

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📁 Upload & Embed PDFs", "🎙️ Ask Question", "📜 Chat History"])

# --- UPLOAD PDFs TAB ---
with tab1:
    st.subheader("📤 Upload PDFs")
    st.markdown("Upload new PDFs here. **Note:** Uploading new files will clear the existing index and embeddings.")
    uploaded_files = st.file_uploader("Select PDF(s)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

    if uploaded_files:
        if st.button("Embed Uploaded PDFs"):
            with st.spinner("Clearing old index and embedding new files..."):
                clear_faiss_index() # Clear previous index
                file_paths = save_uploaded_files(uploaded_files)
                try:
                    success = embed_files_from_paths(file_paths)
                    if success:
                        st.session_state.embedding_created = True
                        st.session_state.qa_chain = None # Reset chain to reload with new retriever
                        st.success("✅ PDFs embedded and saved using FAISS!")
                        st.rerun() # Rerun to update status and load chain
                    else:
                        st.error("⚠️ Embedding failed for an unknown reason.")
                except Exception as e:
                    st.error(f"💥 Error during embedding: {e}")
                    st.session_state.embedding_created = False


# --- LOAD/RELOAD QA CHAIN ---
# Moved chain loading here to ensure it happens after potential embedding
# and uses the latest state of embedding_created
if st.session_state.embedding_created and st.session_state.qa_chain is None:
    try:
        retriever = load_faiss_index()
        st.session_state.qa_chain = get_llm_chain(retriever, st.session_state.memory)
        st.sidebar.success("QA Chain Ready!") # Indicate chain readiness
    except Exception as e:
        st.error(f"❌ Could not load FAISS index or build QA chain: {e}")
        st.session_state.embedding_created = False # Mark as not ready if loading fails


# --- ASK QUESTION TAB ---
with tab2:
    st.subheader("🎤 Ask your question")
    if not st.session_state.embedding_created:
        st.warning("⚠️ Please upload and embed PDFs first in Tab 1.")
    elif st.session_state.qa_chain is None:
         st.warning("⏳ QA Chain is not ready. Please wait or check embedding status.")
    else:
        # Use a form to manage the button press and query processing
        with st.form(key='ask_form'):
            ask_button = st.form_submit_button("🎙️ Press, Speak, then Wait")

            if ask_button:
                st.info("🎧 Listening for your query...")
                # Single call to listen_query
                query = listen_query(timeout=8) # Increased timeout slightly

                if query and not query.startswith("ERROR::") and query.lower().strip() != "goodbye":
                    st.success(f"✅ You said: `{query}`")
                    with st.spinner("🔍 Thinking and Searching..."):
                        try:
                            # Use the QA chain stored in session state
                            result = st.session_state.qa_chain.invoke({"question": query})
                            answer = result.get("answer", "Sorry, I couldn't find an answer.")
                            sources = result.get("source_documents", []) # Langchain 0.1+ uses this key

                            # Append to history BEFORE speaking and displaying
                            st.session_state.chat_history.append((query, answer, sources))

                            # Speak the answer
                            speak_text(answer)

                            # Log the interaction
                            try:
                                with open("chat_log.txt", "a", encoding="utf-8") as f:
                                    f.write(f"\n\n[{datetime.now()}]\nQ: {query}\nA: {answer}\n")
                            except Exception as log_e:
                                st.warning(f"Could not write to chat_log.txt: {log_e}")

                            # Display results within the form context after processing
                            st.success("✅ Answer generated and spoken!")
                            # No automatic rerun here, user stays on the tab

                        except Exception as e:
                            st.error(f"💥 Error during answer generation: {e}")
                            # Attempt to add error indication to history
                            st.session_state.chat_history.append((query, f"Error processing query: {e}", []))

                elif query and query.lower().strip() == "goodbye":
                     st.warning("🎤 Listening stopped. 'Goodbye' detected, but no action taken in single-shot mode.")
                elif query and query.startswith("ERROR::"):
                     # Display the specific Speech Recognition Error
                     st.error(f"⚠️ Speech Recognition Error: {query}")
                     st.session_state.chat_history.append((f"Speech Recognition Attempt Failed", f"Error: {query}", []))
                else:
                    st.warning("⚠️ Could not capture your query or no speech detected.")
                    st.session_state.chat_history.append(("Listening Attempt Failed", "No query captured.", []))

                # Let Streamlit handle rerun implicitly after form submission finishes

# --- CHAT HISTORY TAB ---
with tab3:
    st.subheader("📜 Conversation History")
    if st.session_state.chat_history:
        # Iterate and display newest first
        for i, (q, a, sources) in enumerate(reversed(st.session_state.chat_history), 1):
             # Removed the 'key' argument from st.expander
             with st.expander(f"💬 Turn {len(st.session_state.chat_history) - i + 1}: {q[:50]}..."):
                st.markdown(f"**❓ You Asked:** {q}")
                st.markdown(f"**🧠 Answer:** {a}")
                if sources:
                    st.markdown("**📚 Sources:**")
                    for j, doc in enumerate(sources):
                        # Use source and chunk index for unique key if needed later, but keep simple now
                        with st.container(border=True):
                            st.markdown(f"**Source {j+1}:** `{doc.metadata.get('source', 'Unknown')}` (Page: {doc.metadata.get('page', 'N/A')})")
                            st.caption(doc.page_content[:300] + "...")

    else:
        st.info("💡 No chat history yet. Upload a PDF and ask a question in Tab 2.")