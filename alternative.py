import json
import os
import random
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from gtts import gTTS
from playsound import playsound
import whisper
import subprocess


# Function to speak text using gTTS

def speak(text):
    tts = gTTS(text=text, lang='en', tld='us')
    temp_file = "temp_question.mp3"
    tts.save(temp_file)
    playsound(temp_file)
    os.remove(temp_file)

#Dynamic audio recording

def record_dynamic(
    rcord="output.wav",
    fs=44100,
    silence_limit=5.0,
    block_size=4096
):
    print("🎤 Recording... Speak now.")

    frames_buffer = []
    silence_time = 0.0
    speech_detected = False

    #Levels
    noise_level = None
    speech_level = None
    alpha = 0.30

    #Timings
    hangover_time = 0.8
    hangover_counter = 0.0

    init_blocks = 12   # ~1 sec
    block_count = 0

    def rms(x):
        return np.sqrt(np.mean(x**2))

    def callback(indata, frames, time_info, status):
        nonlocal silence_time, hangover_counter
        nonlocal noise_level, speech_level, speech_detected, block_count

        frames_buffer.append(indata.copy())
        amp = rms(indata.astype(np.float32))
        block_duration = frames / fs

        #Initial noise calibration ONLY
        if not speech_detected and block_count < init_blocks:
            noise_level = amp if noise_level is None else min(noise_level, amp)
            block_count += 1
            return

        #Initialize speech level
        if speech_level is None:
            speech_level = noise_level * 2.5

        threshold = noise_level + alpha * (speech_level - noise_level)

        #Speech detected
        if amp >= threshold:
            speech_detected = True
            silence_time = 0.0
            hangover_counter = 0.0
            speech_level = 0.9 * speech_level + 0.1 * amp
            return

        # Below threshold
        hangover_counter += block_duration

        if hangover_counter < hangover_time:
            return

        if speech_detected:
            silence_time += block_duration

        if silence_time >= silence_limit:
            raise sd.CallbackStop

    with sd.InputStream(
        channels=1,
        samplerate=fs,
        blocksize=block_size,
        dtype="float32",
        callback=callback
    ):
        sd.sleep(30000)

    audio = np.concatenate(frames_buffer, axis=0)
    write(rcord, fs, audio)

    print("Thank you. Recording finished. Processing...")

def remove_silence(input_audio, output_audio):
    command = [
        "ffmpeg", "-y",
        "-i", input_audio,
        "-af", "silenceremove=start_periods=1:start_threshold=-40dB",
        output_audio
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#Loading Whisper model

print("Your exam is beginning now...")
model = whisper.load_model("base")

def audio_to_text(audio_file):
    result = model.transcribe(
        audio_file,
        language="en",
        temperature=0.0,                    
        condition_on_previous_text=False
    )
    return result["text"]

#Cleaning transcribed text

def clean_text(text):
    unwanted = [
        "thank you",
        "thanks for watching",
        "subscribe",
        "um",
        "uh"
    ]
    for word in unwanted:
        text = text.replace(word, "")
    return text.strip()

#Randomly picking a JSON file

questions_folder = "questions"
json_files = [f for f in os.listdir(questions_folder) if f.endswith(".json")]

selected_file = random.choice(json_files)
json_path = os.path.join(questions_folder, selected_file)

print(f"Please answer the questions one by one...")

#Loading all questions from JSON

with open(json_path, "r") as f:
    questions = json.load(f)["questions"]

#Ask questions and record answers

for i, question in enumerate(questions, start=1):

    print(f"\n--- Question {i} ---")
    print(question)

#Speaking question using gTTS

    speak(f"Question {i}. {question}")
    time.sleep(0.3) #small pause before recording

#Record answer dynamically

    raw_audio = f"answer_{i}_raw.wav"
    clean_audio = f"answer_{i}.wav"

    record_dynamic(rcord=raw_audio)
    remove_silence(raw_audio, clean_audio)

#Convert audio to text

    text = audio_to_text(clean_audio)
    text = clean_text(text)

    print(f"\n--- Answer {i} (Text) ---")
    print(text) 