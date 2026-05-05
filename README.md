**IELTS Speaking Module**


**Overview**

This project is an AI-based IELTS Speaking practice system designed to simulate real speaking test conditions. It records user responses, converts speech to text, and provides a structured pipeline for further evaluation and analysis.

The system integrates speech processing techniques with a modular backend, making it suitable for future expansion into a complete IELTS evaluation platform.

**Features**

Voice recording with automatic silence detection
Speech-to-text transcription using Whisper
Text-to-speech prompt generation using gTTS
End-to-end speaking workflow (prompt → response → transcription)
Modular and extensible architecture

**Tech Stack**

Python
FastAPI
OpenAI Whisper
gTTS
sounddevice
NumPy

**Project Structure**

Ielts_speaking/
│── main.py                # Entry point of the application
│── recorder.py            # Handles audio recording and silence detection
│── tts.py                 # Text-to-speech generation
│── transcriber.py         # Speech-to-text using Whisper
│── utils.py               # Utility functions
│── requirements.txt

**Clone the repository:**

git clone https://github.com/SadSaad404/Ielts_speaking.git
cd Ielts_speaking

**Create a virtual environment:**

python -m venv venv

**Activate the environment:**

venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

**Install dependencies:**

pip install -r requirements.txt

**Run the application:**

python main.py

**Workflow:**

1. The system generates or plays a speaking prompt
2. User response is recorded using microphone
3. Silence detection stops recording automatically
4. Audio is transcribed into text using Whisper
5. Output can be used for further evaluation
