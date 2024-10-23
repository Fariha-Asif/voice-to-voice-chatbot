import os
import whisper  # Whisper for transcription
from groq import Groq  # Groq LLM API
from gtts import gTTS  # gTTS for text-to-speech conversion
import gradio as gr  # Gradio for deployment
import tempfile  # Temp files for audio responses

# Load the Whisper model for transcription
whisper_model = whisper.load_model("base")  # Choose model size based on your need

# Set your Groq API key (replace with your actual key)
os.environ['GROQ_API_KEY'] = 'your_groq_api_key_here'

# Function to transcribe audio using the Whisper model
def transcribe_audio(audio_file):
    try:
        print(f"Transcribing audio using Whisper: {audio_file}")
        result = whisper_model.transcribe(audio_file)  # Transcribe using Whisper
        print(f"Whisper transcription result: {result['text']}")
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error during transcription"

# Function to interact with Groq's language model
def chat_with_groq(user_input):
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        print(f"Sending input to Groq LLM: {user_input}")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama3-8b-8192"  # Using Groq's Llama3 model
        )
        print(f"Groq LLM response: {response}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error interacting with Groq LLM: {e}")
        return "Error interacting with LLM"

# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        print(f"Converting text to speech: {text}")
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        print(f"Saved audio response: {temp_file.name}")
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
    inputs=gr.Audio(type="filepath"),  # Use 'filepath' for handling audio input
    outputs=gr.Audio(type="filepath"),  # Return audio response as 'filepath'
    title="Real-time Voice-to-Voice Chatbot",
    description="Speak to the bot, and it will respond with a voice message!"
)

# Launch the Gradio app
interface.launch()
