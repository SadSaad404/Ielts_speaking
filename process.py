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
import ielts_scoring
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
        "Good morning. My name is Ahmed Saad, and I will be conducting your IELTS speaking test today."
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
    total_silence = 0
    long_pauses = 0
    pause_flag = False

    def callback(indata, frames, time_info, status):
        nonlocal recording, silent_time, total_silence, long_pauses, pause_flag
        recording.append(indata.copy())
        volume = np.max(np.abs(indata))
        frame_time = frames / fs

        if volume < threshold:
            silent_time += frame_time
            total_silence += frame_time
            if silent_time >= 0.8 and not pause_flag:
                long_pauses += 1
                pause_flag = True
        else:
            silent_time = 0
            pause_flag = False

    with sd.InputStream(channels=1, samplerate=fs, blocksize=1024, callback=callback):
        while silent_time < silence_limit:
            sd.sleep(100)

    audio_data = np.concatenate(recording, axis=0)
    write(filename, fs, audio_data)

    duration = audio_data.shape[0] / fs
    print(f" Recording saved as {filename}")

    return {
        "duration": duration,
        "silence": total_silence,
        "long_pauses": long_pauses
    }

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



# Ask questions, record answers (EXAM PART)

all_answers = []
total_words = 0
effective_speaking_time = 0
total_long_pauses = 0

for i, question in enumerate(questions, start=1):
    print(f"\nQuestion {i}: {question}")
    speak(f"Question {i}: {question}")
    time.sleep(0.5)

    audio_file = f"answer_{i}.wav"
    metrics = record_dynamic(audio_file)

    text = audio_to_text(audio_file)
    all_answers.append(text)

    words = nltk.word_tokenize(text)
    total_words += len(words)

    speaking_time = metrics["duration"] - metrics["silence"]
    effective_speaking_time += max(speaking_time, 0.1)
    total_long_pauses += metrics["long_pauses"]

    print("\nYour Answer:")
    print(text)


#SCORING

print("Analyzing the results...")

# Use the first answer for analysis (or combine all)
if all_answers:
    main_audio = "answer_1.wav"  # Analyze first answer
    combined_text = " ".join(all_answers)
    
    try:
        # Try to import from ielts_scoring (your file name)
        from ielts_scoring import calculate_band_hybrid, print_hybrid_report
        
        band, details = calculate_band_hybrid(main_audio, combined_text)
        print_hybrid_report(band, details)
        
    except ImportError as e:
        print(f" Hybrid scoring import error: {e}")
        print("Using basic scoring instead...")
        
        # Import your original basic functions
        try:
            from ielts_scoring import calculate_fluency, calculate_coherence, calculate_ttr
            from ielts_scoring import pronunciation_score, calculate_band
        except ImportError:
            # Define them locally if import fails
            def calculate_fluency(total_words, effective_seconds, long_pauses=0):
                if effective_seconds <= 0:
                    return 0
                base = total_words / effective_seconds
                penalty = long_pauses * 0.15
                return max(0, base - penalty)
            
            def calculate_coherence(text):
                linking_words = [
                    "however", "because", "therefore", "although",
                    "but", "so", "and", "then", "also", "for example"
                ]
                sentences = nltk.sent_tokenize(text)
                tokens = nltk.word_tokenize(text.lower())
                count = sum(tokens.count(w) for w in linking_words)
                return count / len(sentences) if sentences else 0
            
            def calculate_ttr(text):
                tokens = nltk.word_tokenize(text.lower())
                return len(set(tokens)) / len(tokens) if tokens else 0
            
            def pronunciation_score():
                return 6.5
            
            def calculate_band(fc_raw, coherence_raw, lexical_raw,
                               grammar_acc, grammar_range_val, pronunciation_val):
                def map_to_band(metric, min_value, max_value):
                    band = 9 * (metric - min_value) / (max_value - min_value)
                    return max(0, min(9, band))
                
                fc_band = map_to_band(fc_raw, 1.5, 4.0)
                coherence_band = map_to_band(coherence_raw, 0, 1)
                lr_band = map_to_band(lexical_raw, 0.3, 0.9)
                gra_band = map_to_band(grammar_acc, 40, 100)
                pron_band = pronunciation_val
                
                avg = (fc_band + coherence_band + lr_band + gra_band + pron_band) / 5
                final_band = round(avg * 2) / 2
                
                return final_band, {
                    "Fluency": round(fc_band, 2),
                    "Coherence": round(coherence_band, 2),
                    "Lexical": round(lr_band, 2),
                    "Grammar": round(gra_band, 2),
                    "Pronunciation": pron_band
                }
        
        full_response = " ".join(all_answers)
        
        fc_raw = total_words / effective_speaking_time if effective_speaking_time > 0 else 0
        coherence_raw = calculate_coherence(full_response)
        lexical_raw = calculate_ttr(full_response)
        
        grammar_acc = 80
        grammar_range_val = 0.5
        pronunciation_val = pronunciation_score()
        
        band, details = calculate_band(
            fc_raw, coherence_raw, lexical_raw,
            grammar_acc, grammar_range_val, pronunciation_val
        )
        
        print(f"\nEstimated Band: {band}")
        print("Breakdown:", details)
else:
    print("No answers to analyze")