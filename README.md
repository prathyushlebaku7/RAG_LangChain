# 🧠 Talk to Your PDFs (Voice-Enabled RAG with FAISS & Mistral)

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Core-lightgrey.svg)](https://python.langchain.com/docs/get_started/introduction)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-blue.svg)](https://github.com/facebookresearch/faiss)
[![Mistral AI](https://img.shields.io/badge/Mistral%20AI-LLM-orange.svg)](https://mistral.ai/)

---

**Interact with your PDF documents using your voice!** This application leverages Retrieval-Augmented Generation (RAG) to allow you to ask questions about uploaded PDFs and receive spoken answers. It uses FAISS for efficient similarity search, Hugging Face sentence transformers for embeddings, Mistral AI for language understanding and generation, and Streamlit for the user interface.

![image](https://github.com/user-attachments/assets/23ae0465-76cf-4e94-86f4-c1080801dd8e))

---

## ✨ Features

* **📁 PDF Upload:** Upload one or multiple PDF documents.
* **⚙️ Automatic Embedding:** Uploaded PDFs are automatically processed, chunked, and embedded using `sentence-transformers/all-MiniLM-L6-v2`.
* **💾 FAISS Vector Store:** Embeddings are stored locally in a FAISS index for fast retrieval.
* **🎙️ Voice Input:** Ask questions using your microphone.
* **🔊 Spoken Answers:** Receive answers spoken back to you using text-to-speech.
* **🧠 RAG Implementation:** Uses LangChain and Mistral AI (`mistral-small-latest`) to generate answers based on the content of your documents.
* **📜 Chat History:** View your conversation history with sources for each answer.
* **🔄 State Management:** Remembers uploaded PDFs and chat history within a session.

---

## 🛠️ Technology Stack

* **Frontend:** Streamlit
* **LLM & RAG Framework:** LangChain, LangChain MistralAI
* **Language Model:** Mistral AI (`mistral-small-latest`)
* **Embedding Model:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Store:** FAISS (Facebook AI Similarity Search) - CPU version
* **PDF Processing:** PyMuPDF
* **Speech-to-Text:** SpeechRecognition (uses Google Web Speech API by default)
* **Text-to-Speech:** pyttsx3
* **Core Language:** Python

---

## 📂 File Structure


RAG_LangChain/
│
├── app.py                # Main Streamlit application, UI, and flow control
├── embeddings.py         # Handles PDF loading, chunking, embedding, FAISS storage/loading
├── generation.py         # Handles LLM chain, Speech-to-Text, Text-to-Speech
├── requirements.txt      # Python dependencies
├── faiss_index/          # Directory created to store the FAISS index (auto-generated)
├── uploaded_pdfs/        # Directory created to store uploaded PDFs temporarily (auto-generated)
└── chat_log.txt          # Log file for conversations (auto-generated)


---

## 🚀 Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/RAG_LangChain.git](https://github.com/your-username/RAG_LangChain.git) # Replace with your repo URL
    cd RAG_LangChain
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv rag_venv
    # Activate it:
    # Windows: .\rag_venv\Scripts\activate
    # macOS/Linux: source rag_venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **Note:** You might need system libraries for `pyttsx3` (like `espeak` or `nsss`) and `PortAudio` (for `PyAudio`, which `SpeechRecognition` often uses). Consult the documentation for these libraries if you encounter installation issues.

4.  **Configure API Key:**
    * This project requires a Mistral AI API key.
    * **IMPORTANT:** The key is currently **hardcoded** in `generation.py`. For security, it's **highly recommended** to modify the code to load the key from an environment variable or a `.env` file instead.
    * Example using environment variable:
        * Set an environment variable: `export MISTRAL_API_KEY='your_mistral_api_key'` (macOS/Linux) or `set MISTRAL_API_KEY=your_mistral_api_key` (Windows CMD) or `$env:MISTRAL_API_KEY='your_mistral_api_key'` (Windows PowerShell).
        * Modify `generation.py` to load it:
            ```python
            import os
            from dotenv import load_dotenv # pip install python-dotenv

            load_dotenv() # Load .env file if present
            mistral_api_key = os.getenv("MISTRAL_API_KEY")

            # ... later in get_llm_chain ...
            llm = ChatMistralAI(
                model="mistral-small-latest",
                api_key=mistral_api_key, # Use the loaded key
                temperature=0.7
            )
            ```

---

## ▶️ How to Run

1.  Ensure your virtual environment is activated.
2.  Make sure you have configured your Mistral API key (see Setup step 4).
3.  Navigate to the project directory in your terminal.
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
5.  The application should open in your default web browser.

---

## 💡 Usage

1.  **Upload PDFs:** Go to the "📁 Upload & Embed PDFs" tab and upload one or more PDF files. Click "Embed Uploaded PDFs". Wait for the embedding process to complete.
2.  **Ask Question:** Go to the "🎙️ Ask Question" tab. Click the "🎙️ Press, Speak, then Wait" button. Speak your question clearly into your microphone.
3.  **Get Answer:** The application will transcribe your speech, query the LLM using the relevant document chunks, generate an answer, and speak it back to you.
4.  **View History:** Go to the "📜 Chat History" tab to see previous questions, answers, and the source document chunks used.
5.  **Clear History:** Use the button in the sidebar to clear the conversation history and LangChain memory if needed.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
*(Optional: Add more specific contribution guidelines if desired)*

---

## 📄 License

*(Suggestion: Add a license file (e.g., MIT, Apache 2.0) to your repository and specify it here.)*
