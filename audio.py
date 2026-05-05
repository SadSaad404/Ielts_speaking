import json
from gtts import gTTS
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import time
import os
import random
from playsound import playsound

# Function to speak text using gTTS

def speak(text):
    tts = gTTS(text=text, lang='en',tld='us')
    rcord = "temp_question.mp3"
    tts.save(rcord)
    playsound(rcord)
    os.remove(rcord)

#Dynamic audio recording

def record_dynamic(rcord="output.wav", fs=44100, silence_limit=7, threshold=0.015, block_size=4096):
    
    print("Recording... Speak now.")
    recording = []
    continuous_silence = 0

    def callback(indata, frames, time, status):
        nonlocal recording, continuous_silence
        recording.append(indata.copy())
        amplitude = np.max(np.abs(indata))
        if amplitude < threshold:
            continuous_silence += frames / fs
        else:
            continuous_silence = 0

    with sd.InputStream(channels=1, samplerate=fs, blocksize=block_size, callback=callback):
        while continuous_silence < silence_limit:
            sd.sleep(100)

    audio_data = np.concatenate(recording, axis=0)
    write(rcord, fs, audio_data)
    print(f"Recording finished. Saved as {rcord}")

#Loading Whisper model

print("Loading Whisper model...")
model = whisper.load_model("base")

def audio_to_text(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

#Randomly picking a JSON file

questions_folder = "questions"
json_files = [f for f in os.listdir(questions_folder) if f.endswith(".json")]
selected_file = random.choice(json_files)
json_path = os.path.join(questions_folder, selected_file)
print(f"Selected question file: {selected_file}")

#Loading all questions from JSON

with open(json_path, "r") as f:
    data = json.load(f)
questions = data["questions"]

#Ask questions and record answers

for i, question in enumerate(questions, start=1):
    print(f"\n--- Question {i} ---")
    print(question)

#Speaking question using gTTS

    speak(f"Question {i}: {question}")
    time.sleep(0.1)  # small pause before recording

#Record answer dynamically

    audio_file = f"answer_{i}.wav"
    record_dynamic(rcord=audio_file)

#Convert audio to text
    answer_text = audio_to_text(audio_file)

    print(f"\n--- Answer {i} (Text) ---")
    print(answer_text)