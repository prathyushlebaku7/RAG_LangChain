import pyttsx3
import speech_recognition as sr
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI
import threading

def get_llm_chain(retriever, memory):
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key="2WIkmsiJYxWE610QFydCaA5LJJIyPl0h",
        temperature=0.7
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

def speak_text(text):
    def run_tts():
        try:
            safe_text = text.encode('utf-8', 'replace').decode('utf-8')
        except Exception:
            safe_text = "Error reading response"
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(safe_text)
        engine.runAndWait()
        engine.stop()
        del engine

    tts_thread = threading.Thread(target=run_tts)
    tts_thread.start()
    tts_thread.join()

def listen_query(timeout=6):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("🎤 Listening for your query...")
            audio = recognizer.listen(source, timeout=timeout)
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected. Ending loop.")
            return "goodbye"
        except Exception as e:
            return f"ERROR::{e}"
