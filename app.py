from flask import Flask, render_template, request, jsonify
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import pandas as pd
import time
import random

app = Flask(__name__)

# ---------------------------
# Config
# ---------------------------
audio_folder = "audios"
books_file = "books_with_clean_genres.csv"
sample_rate = 44100
channels = 1

os.makedirs(audio_folder, exist_ok=True)

# ---------------------------
# Record Audio
# ---------------------------
def record_audio(filename, duration):
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
        audio_data = np.clip(recording, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        if audio_int16.shape[1] == 1:
            audio_to_write = audio_int16[:, 0]
        else:
            audio_to_write = audio_int16
        write(filename, sample_rate, audio_to_write)
        return True
    except Exception as e:
        print("Recording error:", e)
        return False

# ---------------------------
# Speech to Text
# ---------------------------
def audio_to_text(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text
    except:
        return ""

# ---------------------------
# Genre Extraction
# ---------------------------
def extract_genre(text):
    genre_keywords = {
        "mystery": "Thriller",
        "crime": "Thriller",
        "detective": "Thriller",
        "suspense": "Thriller",
        "history": "Non-fiction",
        "historical": "Non-fiction",
        "self-help": "Self-help",
        "motivation": "Self-help",
        "memoir": "Memoir",
        "biograph": "Memoir",
        "fiction": "Fiction",
        "fantasy": "Fiction",
        "science": "Non-fiction",
        "classic": "Classic",
        "romance": "Fiction",
        "adventure": "Fiction",
        "romantic": "Fiction",
        "thriller": "Thriller",
        "humour": "Humor",
        "comedy": "Comedy"
    }
    detected = []
    lower = text.lower()
    for k, v in genre_keywords.items():
        if k in lower:
            detected.append(v)
    return list(dict.fromkeys(detected))  # remove duplicates while keeping order

# ---------------------------
# Recommend Books (FIXED)
# ---------------------------
def recommend_books(genres):
    try:
        books_df = pd.read_csv(books_file)
    except Exception as e:
        print("Books file read error:", e)
        return []

    # Convert genres column to lowercase for flexible matching
    books_df['genres'] = books_df['genres'].astype(str).str.lower()

    matched_books = pd.DataFrame()

    # Match any detected genre partially inside the genres column
    for g in genres:
        matches = books_df[books_df['genres'].str.contains(g.lower(), na=False)]
        matched_books = pd.concat([matched_books, matches])

    # Remove duplicates
    matched_books = matched_books.drop_duplicates(subset=['Title', 'Author'])

    if matched_books.empty:
        print("No books found for genres:", genres)
        return []

    # Shuffle and return top 5
    matched_books = matched_books.sample(frac=1).head(5)

    return matched_books.to_dict(orient="records")

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/record", methods=["POST"])
def record():
    duration = float(request.form.get("duration", 5))
    user = request.form.get("user", "guest")
    filename = os.path.join(audio_folder, f"{user}_{int(time.time())}.wav")

    success = record_audio(filename, duration)
    if not success:
        return jsonify({"error": "Recording failed"})

    text = audio_to_text(filename)
    genres = extract_genre(text)
    recs = recommend_books(genres) if genres else []

    return jsonify({
        "transcription": text,
        "genres": genres,
        "recommendations": recs
    })

if __name__ == "__main__":
    app.run(debug=True)
