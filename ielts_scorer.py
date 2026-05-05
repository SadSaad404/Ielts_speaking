
import nltk
from collections import Counter

# Utility

def tokenize(text):
    return nltk.word_tokenize(text.lower())

# 1. Fluency & Coherence

def calculate_fluency(total_words, total_seconds, repetitions=0, self_corrections=0):
    if total_seconds == 0:
        return 0
    fluency_raw = (total_words - (repetitions + self_corrections)) / total_seconds
    return fluency_raw

def calculate_coherence(text):
    linking_words = [
        "however", "because", "therefore", "although", "but",
        "so", "and", "then", "also", "for example"
    ]
    sentences = nltk.sent_tokenize(text)
    tokens = tokenize(text)
    count = sum(tokens.count(word) for word in linking_words)
    return count / len(sentences) if sentences else 0


# 2. Lexical Resource

def calculate_ttr(text):
    tokens = tokenize(text)
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

def repetition_rate(text):
    tokens = tokenize(text)
    freq = Counter(tokens)
    repeated = sum(v-1 for v in freq.values() if v>1)
    return repeated / len(tokens) if tokens else 0


# 3. Grammar

def grammar_accuracy(error_free_clauses, total_clauses):
    return (error_free_clauses / total_clauses) * 100 if total_clauses else 0

def grammar_range(complex_sentences, total_sentences):
    return complex_sentences / total_sentences if total_sentences else 0

# 4. Pronunciation (Placeholder)

def pronunciation_score():
    return 6.5


# Mapping raw metrics to IELTS 0-9 scale

def map_to_band(metric, min_value, max_value):
    """
    Linear map: min_value → 0, max_value → 9
    Clipped between 0-9
    """
    band = 9 * (metric - min_value) / (max_value - min_value)
    return max(0, min(9, band))


# Band calculation

def calculate_band(fc_raw, coherence_raw, lexical_raw, grammar_acc, grammar_range_val, pronunciation_val):
    """
    Calculate IELTS-style band score (0-9)
    """

    # Map raw metrics to 0-9 scale
    fc_band = map_to_band(fc_raw, min_value=1, max_value=4)  # words/sec ~ 1-4
    coherence_band = map_to_band(coherence_raw, min_value=0, max_value=1)
    lr_band = map_to_band(lexical_raw, min_value=0.3, max_value=0.9)
    gra_band = map_to_band(grammar_acc, min_value=40, max_value=100)
    pronunciation_band = pronunciation_val  # already 0-9

    # Average each category
    average = (fc_band + coherence_band + lr_band + gra_band + pronunciation_band) / 5

    # Round to nearest 0.5 (IELTS convention)
    band_score = round(average * 2) / 2
    return band_score, {
        "Fluency": fc_band,
        "Coherence": coherence_band,
        "Lexical": lr_band,
        "Grammar": gra_band,
        "Pronunciation": pronunciation_band
    }