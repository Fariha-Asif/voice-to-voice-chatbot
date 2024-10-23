import os
import whisper  # Import whisper for local transcription
from groq import Groq
from gtts import gTTS
import gradio as gr
import tempfile

# Load the whisper model
whisper_model = whisper.load_model("base")  # You can also use "small", "medium", "large" based on your preference

# Set your API key for Groq
os.environ['GROQ_API_KEY'] = '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'

# Function to transcribe audio using the local Whisper model
def transcribe_audio(audio_file):
    try:
        print(f"Transcribing audio using Whisper: {audio_file}")  # Debugging
        # Perform transcription using the whisper model
        result = whisper_model.transcribe(audio_file)
        print(f"Whisper transcription result: {result['text']}")  # Debugging
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error during transcription"

# Function to interact with Groq's language model
def chat_with_groq(user_input):
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        print(f"Sending input to Groq LLM: {user_input}")  # Debugging print
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama3-8b-8192"
        )
        print(f"Groq LLM response: {response}")  # Debugging
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error interacting with Groq LLM: {e}")
        return "Error interacting with LLM"

# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        print(f"Converting text to speech: {text}")  # Debugging
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        print(f"Saved audio response: {temp_file.name}")  # Debugging
        return temp_file.name
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return None

# Main function to process audio input, get LLM response, and return audio output
def process_audio(audio):
    try:
        # Step 1: Transcribe the audio input
        transcription = transcribe_audio(audio)
        if transcription.startswith("Error"):
            return None
        
        # Step 2: Get the chatbot response from Groq's LLM
        response = chat_with_groq(transcription)
        if response.startswith("Error"):
            return None

        # Step 3: Convert the chatbot response to speech
        audio_response = text_to_speech(response)
        if not audio_response:
            return None

        return audio_response
    except Exception as e:
        print(f"Error in process_audio: {e}")
        return None

# Gradio interface setup
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),  # Using 'filepath' for file handling
    outputs=gr.Audio(type="filepath"), # Using 'filepath' for file handling
    title="Real-time Voice-to-Voice Chatbot",
    description="Speak to the bot, and it will respond with a voice message!"
)

# Launch the Gradio app
interface.launch()
