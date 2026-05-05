import json
import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import time
import random
import sys
import os
import nltk

# Function to speak text using gTTS

def speak(text):
    async def _speak():
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        temp_file = "temp_question.mp3"
        
        # Save audio
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                with open(temp_file, "ab") as f:
                    f.write(chunk["data"])
        
        # Play with ffplay (you have FFmpeg)
        import subprocess
        subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_file], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_file)
    
    asyncio.run(_speak())

#intro

def examiner_introduction():
    intro_lines = [
        "Good morning. My name is Innovative skills, and I will be conducting your IELTS speaking test today.",
        "Can you tell me your full name, please?",
        "Thank you. And what shall I call you?",
        "Can you show me your identification, please?",
        "Thank you. Now we will begin with Part One."
    ]

    for line in intro_lines:
        print(f"\nExaminer: {line}")
        speak(line)
        time.sleep(0.5)

# Dynamic audio recording

def record_dynamic(filename="output.wav", fs=44100, silence_limit=5, threshold=0.064):
    print(" Recording... Speak now (stop talking for 5 seconds to finish)")
    recording = []
    silent_time = 0
    
    def callback(indata, frames, time_info, status):
        nonlocal recording, silent_time
        recording.append(indata.copy())
        volume = np.max(np.abs(indata))
        
        if volume < threshold:
            silent_time += frames / fs
        else:
            silent_time = 0
    
    with sd.InputStream(channels=1, samplerate=fs, blocksize=1024, callback=callback):
        while silent_time < silence_limit:
            sd.sleep(100)
    
    audio_data = np.concatenate(recording, axis=0)
    write(filename, fs, audio_data)
    print(f" Recording saved as {filename}")

# Load Whisper model

print("Loading Whisper model...")
model = whisper.load_model("medium")

def audio_to_text(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# Look for JSON files in 'questions' folder

questions_folder = "questions"
print(f"\nLooking for question files in '{questions_folder}' folder...")

# Check if questions folder exists

if not os.path.exists(questions_folder):
    print(f" Folder '{questions_folder}' not found!")
    print(f"Creating '{questions_folder}' folder...")
    os.makedirs(questions_folder)
    print(f" Created '{questions_folder}' folder.")
    print(f"Please add your JSON files (part1_set1.json, etc.) to this folder.")
    exit()

# Look for JSON files in the questions folder

json_files = [f for f in os.listdir(questions_folder) 
              if f.endswith('.json') and f.startswith('part1_set')]

# Check if we found any files

if not json_files:
    print(f" No JSON files found in '{questions_folder}' folder")
    print(f"Files in folder: {os.listdir(questions_folder)}")
    print("Make sure you have: part1_set1.json, part1_set2.json, part1_set3.json")
    exit()

print(f"Found {len(json_files)} JSON files: {json_files}")

# Randomly pick a file

selected_file = random.choice(json_files)
json_path = os.path.join(questions_folder, selected_file)
print(f"\nSelected question file: {selected_file}")

# Load questions from JSON

with open(json_path, "r") as f:
    data = json.load(f)

# Try different possible keys for questions

if "questions" in data:
    questions = data["questions"]
elif "questions_list" in data:
    questions = data["questions_list"]
elif "part1_questions" in data:
    questions = data["part1_questions"]
elif isinstance(data, list):
    # If the JSON file is directly an array
    questions = data
else:
# If no standard key, take the first list we find

    for key in data:
        if isinstance(data[key], list):
            questions = data[key]
            break
    else:
        print("Could not find questions in the JSON file")
        print(f"JSON structure: {type(data)}")
        if isinstance(data, dict):
            print(f"Available keys: {list(data.keys())}")
        exit()

print(f"Found {len(questions)} questions in the file")

# Shuffle questions to ask random ones
random.shuffle(questions)

# Ask only 3 questions (adjust as needed)
num_questions = min(3, len(questions))
questions_to_ask = questions[:num_questions]

examiner_introduction()
print(f"\nI will ask you {num_questions} random questions from the set.\n")

# Ask questions, record answers

for i, question in enumerate(questions_to_ask, start=1):
    print(f"Question {i}/{num_questions}")
    print(f" {question}")
    
    # Speak question
    speak(f"Question {i}: {question}")
    
    # Small pause before recording
    time.sleep(0.5)
    
    # Record answer
    audio_file = f"answer_{i}.wav"
    record_dynamic(filename=audio_file)
    
    # Convert audio to text
    answer_text = audio_to_text(audio_file)
    
    print(f"\n Your Answer:")
    print(answer_text if answer_text else "(No speech detected)")
    
    # Small pause before next question
    if i < num_questions:
        print("\nNext question in 3 seconds...")
        time.sleep(3)

    print(f"\n--- Answer {i} (Text) ---")
    print(answer_text)