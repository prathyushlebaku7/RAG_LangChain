import streamlit as st
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from embeddings import clear_faiss_index, save_uploaded_files, embed_files_from_paths, load_faiss_index
from generation import (
    get_llm_chain,
    speak_text,
    listen_query,
    LISTENING_TIMEOUT,
    STOP_COMMAND_RECEIVED,
    SPEECH_RECOGNITION_ERROR # Note: This is SPEECH_RECOGNITION_ERROR_PREFIX in generation.py
)
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ask Your PDFs", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠 Talk to Your PDFs (FAISS Version)</h1>", unsafe_allow_html=True)

# --- CUSTOM STYLES ---
st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stButton>button {
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        /* Custom button colors for specific keys or classes if possible */
        /* Streamlit's `type` attribute is preferred: type="primary" or type="secondary" */
        /* For more specific styling, you might need to target generated classes or use st.markdown with HTML */
        
        /* Example specific styling for start/stop buttons if type attribute isn't enough */
        /* button[data-testid="stButton-Start Voice Chat"], /* Streamlit <1.30 */
        /* button[data-testid="stButton-start_chat_button"] { /* Streamlit >=1.30 for key */
        /* background-color: #4CAF50 !important;
        /* color: white !important;
        /* }
        /* button[data-testid="stButton-Stop Voice Chat"],
        /* button[data-testid="stButton-stop_chat_button"] {
        /* background-color: #f44336 !important;
        /* color: white !important;
        /* } */

        .stExpanderHeader { font-size: 18px !important; }
        .stSpinner > div > div { border-top-color: #4CAF50; }

        .chat-bubble {
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 8px;
            max-width: 75%;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #DCF8C6; /* Light green, like WhatsApp user */
            margin-left: auto;
            text-align: right;
            border-bottom-right-radius: 0px;
        }
        .bot-bubble {
            background-color: #E0E0E0; /* Light grey */
            margin-right: auto;
            text-align: left;
            border-bottom-left-radius: 0px;
        }
        .status-info {
            font-style: italic;
            color: #555;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
if "embedding_created" not in st.session_state:
    st.session_state.embedding_created = os.path.exists("./faiss_index")
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
# --- Voice Loop Specific Session States ---
if "listening_active" not in st.session_state:
    st.session_state.listening_active = False
if "user_query_for_processing" not in st.session_state: # Holds query between listen and process steps
    st.session_state.user_query_for_processing = None
if "status_message" not in st.session_state: # To display messages like "Listening...", "Processing..."
    st.session_state.status_message = ""
if "speak_next_answer" not in st.session_state: # Flag to control speaking
    st.session_state.speak_next_answer = ""


# --- SIDEBAR STATUS ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.markdown(f"**✅ Embeddings Ready:** {'Yes' if st.session_state.embedding_created else 'No'}")
    st.markdown(f"**🎙️ Voice Chat Active:** {'<span style=''color:green;''>Yes</span>' if st.session_state.listening_active else '<span style=''color:red;''>No</span>'}", unsafe_allow_html=True)
    st.markdown(f"**💬 Chat Turns:** `{len(st.session_state.chat_history)}`")
    if st.button("Clear Chat History & Memory", key="clear_history_sidebar"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.session_state.listening_active = False
        st.session_state.user_query_for_processing = None
        st.session_state.status_message = "Chat history cleared. Voice chat stopped."
        st.session_state.speak_next_answer = ""
        st.rerun()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📁 Upload & Embed PDFs", "🎙️ Ask Question", "📜 Chat History"])

# --- UPLOAD PDFs TAB ---
with tab1:
    st.subheader("📤 Upload PDFs")
    st.markdown(
        "Upload new PDFs here. **Note:** Uploading new files will clear the existing index and embeddings."
    )
    uploaded_files = st.file_uploader(
        "Select PDF(s)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_files:
        if st.button("Embed Uploaded PDFs", type="primary", key="embed_button"):
            with st.spinner("Clearing old index and embedding new files..."):
                clear_faiss_index()
                file_paths = save_uploaded_files(uploaded_files)
                try:
                    success = embed_files_from_paths(file_paths)
                    if success:
                        st.session_state.embedding_created = True
                        st.session_state.qa_chain = None  # Reset chain to reload
                        st.session_state.status_message = "✅ PDFs embedded successfully!"
                        st.rerun()
                    else:
                        st.error("⚠️ Embedding failed for an unknown reason.")
                        st.session_state.status_message = "⚠️ Embedding failed."
                except Exception as e:
                    st.error(f"💥 Error during embedding: {e}")
                    st.session_state.embedding_created = False
                    st.session_state.status_message = f"💥 Error during embedding: {e}"


# --- LOAD/RELOAD QA CHAIN ---
# This logic runs on every rerun if conditions are met
if st.session_state.embedding_created and st.session_state.qa_chain is None:
    st.sidebar.info("Loading QA Chain...")
    try:
        retriever = load_faiss_index()
        st.session_state.qa_chain = get_llm_chain(retriever, st.session_state.memory)
        st.sidebar.success("QA Chain Ready!")
    except Exception as e:
        st.sidebar.error(f"❌ QA chain load error: {e}")
        st.session_state.embedding_created = False # Mark as not ready
        st.session_state.listening_active = False # Stop listening if chain fails

# --- ASK QUESTION TAB (VOICE LOOP) ---
with tab2:
    st.subheader("💬 Voice Conversation")

    if not st.session_state.embedding_created:
        st.warning("⚠️ Please upload and embed PDFs first in Tab 1.")
    elif st.session_state.qa_chain is None:
        st.warning("⏳ QA Chain is not ready. Please wait or check Tab 1 if embedding failed.")
    else:
        # Voice Chat Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.listening_active:
                if st.button("🎙️ Start Voice Chat", type="primary", key="start_chat_button", use_container_width=True):
                    st.session_state.listening_active = True
                    st.session_state.status_message = "🎤 Voice chat started. Listening..."
                    st.session_state.user_query_for_processing = None
                    st.session_state.speak_next_answer = "" # Clear any pending speech
                    st.rerun()
        with col2:
            if st.session_state.listening_active:
                if st.button("🛑 Stop Voice Chat", type="secondary", key="stop_chat_button", use_container_width=True):
                    st.session_state.listening_active = False
                    st.session_state.status_message = "🛑 Voice chat stopped by user."
                    st.session_state.user_query_for_processing = None
                    # Optionally speak confirmation:
                    # st.session_state.speak_next_answer = "Voice chat ended."
                    st.rerun()

        # Display current status message
        if st.session_state.status_message:
            st.markdown(f"<p class='status-info'>{st.session_state.status_message}</p>", unsafe_allow_html=True)


        # 1. SPEAK PENDING ANSWER (if any from previous run)
        if st.session_state.speak_next_answer:
            text_to_speak = st.session_state.speak_next_answer
            st.session_state.speak_next_answer = "" # Clear after getting it
            speak_text(text_to_speak)
            # No rerun here, let it flow to listening if active

        # 2. LISTENING PHASE (if active and no query is pending processing)
        if st.session_state.listening_active and not st.session_state.user_query_for_processing and not st.session_state.speak_next_answer :
            status_placeholder = st.empty()
            status_placeholder.info("🎤 Listening... Say 'stop listening' or 'goodbye' to end.")

            query_result = listen_query(timeout=7, phrase_time_limit=12)
            status_placeholder.empty() # Clear "Listening..."

            if query_result == LISTENING_TIMEOUT:
                st.session_state.status_message = "⏱️ Didn't catch that. Try speaking again."
                # Potentially speak this message too
                # st.session_state.speak_next_answer = "I didn't catch that."
                st.rerun()
            elif query_result == STOP_COMMAND_RECEIVED:
                st.session_state.listening_active = False
                st.session_state.status_message = "🛑 Voice chat ended by command."
                st.session_state.speak_next_answer = "Okay, ending voice chat."
                st.rerun()
            elif query_result and query_result.startswith(SPEECH_RECOGNITION_ERROR): # Check prefix
                error_msg = query_result.split("::", 1)[1] if "::" in query_result else "Unknown speech error"
                st.session_state.status_message = f"⚠️ Speech Recognition Error: {error_msg}. Try again."
                st.session_state.chat_history.append( (f"SR Attempt Failed", f"Error: {error_msg}", []) )
                st.session_state.speak_next_answer = f"I had a speech recognition error: {error_msg}. Please try again."
                st.rerun()
            elif query_result: # Valid query received
                st.session_state.user_query_for_processing = query_result
                st.session_state.status_message = f"✅ You said: \"{query_result}\". Processing..."
                st.rerun() # Rerun to move to processing phase
            else: # Should ideally not be reached if listen_query is robust
                st.session_state.status_message = "⚠️ No query captured. Listening again..."
                st.rerun()


        # 3. PROCESSING PHASE (if a query is pending and listening is still active)
        if st.session_state.user_query_for_processing and st.session_state.listening_active:
            query = st.session_state.user_query_for_processing
            st.session_state.user_query_for_processing = None # Consume the query

            # Display user query immediately
            st.markdown(f"<div class='chat-bubble user-bubble'>You: {query}</div>", unsafe_allow_html=True)

            with st.spinner(f"🔍 Thinking about: \"{query}\"..."):
                try:
                    result = st.session_state.qa_chain.invoke({"question": query})
                    answer = result.get("answer", "Sorry, I couldn't find an answer to that.")
                    sources = result.get("source_documents", [])

                    st.session_state.chat_history.append((query, answer, sources))
                    st.session_state.speak_next_answer = answer # Queue answer for speaking
                    st.session_state.status_message = f"💡 Answer found for \"{query}\". Preparing to listen for next query..."

                    # Display bot answer
                    st.markdown(f"<div class='chat-bubble bot-bubble'>Bot: {answer}</div>", unsafe_allow_html=True)

                    # Log to file
                    try:
                        with open("chat_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"\n\n[{datetime.now()}]\nQ: {query}\nA: {answer}\n")
                    except Exception as log_e:
                        st.warning(f"Could not write to chat_log.txt: {log_e}")

                except Exception as e:
                    st.error(f"💥 Error during answer generation: {e}")
                    answer = f"Sorry, an error occurred while processing your question: {str(e)[:100]}"
                    st.session_state.chat_history.append((query, answer, []))
                    st.session_state.speak_next_answer = answer # Queue error message for speaking
                    st.session_state.status_message = f"💥 Error processing \"{query}\". Listening again."

            st.rerun() # Rerun to speak the answer and then listen again

# --- CHAT HISTORY TAB ---
with tab3:
    st.subheader("📜 Conversation History")
    if st.button("Refresh History", key="refresh_history_tab3"):
        st.rerun()

    if st.session_state.chat_history:
        for i, (q, a, sources) in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 Turn {len(st.session_state.chat_history) - i}: {q[:50]}..."):
                st.markdown(f"<div class='chat-bubble user-bubble' style='text-align:left; margin-left:0;'><b>You:</b> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble bot-bubble' style='text-align:left; margin-right:0;'><b>Bot:</b> {a}</div>", unsafe_allow_html=True)
                if sources:
                    st.markdown("--- \n**📚 Sources:**")
                    for j, doc in enumerate(sources):
                        with st.container(border=True):
                            source_name = doc.metadata.get('source', 'Unknown Source')
                            page_num = doc.metadata.get('page', 'N/A')
                            st.markdown(f"**File:** `{os.path.basename(source_name)}` (Page: {page_num})")
                            st.caption(doc.page_content[:350] + "...")
    else:
        st.info("💡 No chat history yet. Upload a PDF and start a voice chat in Tab 2.")
