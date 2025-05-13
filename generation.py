import pyttsx3
import speech_recognition as sr
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI
import threading

# Constants for listen_query results
LISTENING_TIMEOUT = "LISTENING_TIMEOUT"
STOP_COMMAND_RECEIVED = "STOP_COMMAND_RECEIVED"
SPEECH_RECOGNITION_ERROR = "SPEECH_RECOGNITION_ERROR_PREFIX" # Prefix to append specific error

def get_llm_chain(retriever, memory):
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key="2WIkmsiJYxWE610QFydCaA5LJJIyPl0h", # REMINDER: Consider using Streamlit secrets or env variables
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
            # Attempt to clean text for TTS
            safe_text = ''.join(char for char in text if char.isprintable())
            engine = pyttsx3.init()
            engine.setProperty('rate', 170) # Adjust rate as needed
            # Optional: Get available voices and set one if desired
            # voices = engine.getProperty('voices')
            # engine.setProperty('voice', voices[0].id) # Example: set to the first available voice
            engine.say(safe_text)
            engine.runAndWait()
        except RuntimeError as e:
            print(f"TTS pyttsx3 RuntimeError: {e}. Often related to re-initialization or event loop issues.")
            # Try to recover or simply skip speaking if engine is busy
            try:
                engine.stop() # Attempt to stop any previous run
            except Exception as stop_e:
                print(f"TTS: Error trying to stop engine: {stop_e}")
        except Exception as e:
            print(f"TTS General Error: {e}")
        finally:
            # It's tricky to manage engine.stop() and del engine here
            # if the instance is meant to be reused or if errors occur during init.
            # For single use in a thread, stopping might be okay.
            # If you encounter "AttributeError: 'NoneType' object has no attribute 'runAndWait'"
            # it might mean engine failed to initialize.
            pass # Let thread end


    tts_thread = threading.Thread(target=run_tts)
    tts_thread.start()
    tts_thread.join() # This still makes the main Streamlit thread wait for TTS to complete.

def listen_query(timeout=7, phrase_time_limit=12): # Increased timeouts slightly
    recognizer = sr.Recognizer()
    # You might want to adjust energy_threshold dynamically if ambient noise is an issue
    # recognizer.energy_threshold = 4000 # Example: adjust this value
    # recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        try:
            print(f"🎤 Adjusting for ambient noise (1 sec)...")
            recognizer.adjust_for_ambient_noise(source, duration=1) # Adjust for ambient noise
            print(f"🎤 Listening for your query (timeout: {timeout}s, phrase_limit: {phrase_time_limit}s)...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            query = recognizer.recognize_google(audio).lower().strip()
            print(f"✅ You said: {query}") # For debugging

            stop_commands = ["goodbye", "stop listening", "exit chat", "end chat", "terminate session"]
            if query in stop_commands:
                return STOP_COMMAND_RECEIVED
            if not query: # If recognizer returns empty string
                return LISTENING_TIMEOUT # Treat as no meaningful input
            return query
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected within timeout.")
            return LISTENING_TIMEOUT
        except sr.UnknownValueError:
            print("❓ Google Speech Recognition could not understand audio")
            return SPEECH_RECOGNITION_ERROR + "::Could not understand audio"
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return SPEECH_RECOGNITION_ERROR + f"::Service error - {e}"
        except Exception as e:
            print(f"An unexpected error occurred during speech recognition: {e}")
            return SPEECH_RECOGNITION_ERROR + f"::Unexpected SR error - {e}"
