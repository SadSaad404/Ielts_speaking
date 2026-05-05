import nltk
import re
import math
import numpy as np
from collections import Counter
from parselmouth import Sound

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

def tokenize(text):
    return nltk.word_tokenize(text.lower())

def sent_tokenize(text):
    return nltk.sent_tokenize(text)

# PRAAT-BASED PRONUNCIATION & FLUENCY 

def analyze_with_praat(audio_file):
    """
    Extract Praat features for pronunciation and fluency
    Returns: (pronunciation_score, fluency_score, details)
    """
    try:
        sound = Sound(audio_file)
        duration = sound.get_total_duration()
        
        # PRONUNCIATION FEATURES
        
        # 1. Formant dispersion (vowel clarity)
        try:
            formant = sound.to_formant_burg()
            times = formant.xs()[:50]  # First 50 time points
            
            f1_vals, f2_vals = [], []
            for t in times:
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                if f1 and f2:
                    f1_vals.append(f1)
                    f2_vals.append(f2)
            
            if f1_vals and f2_vals:
                f1_var = np.std(f1_vals)
                f2_var = np.std(f2_vals)
                # Clear pronunciation has moderate formant variation
                clarity = 1 / (1 + (f1_var + f2_var) / 1000)
            else:
                clarity = 0.5
        except:
            clarity = 0.5
        
        # 2. Pitch variation (intonation)
        try:
            pitch = sound.to_pitch()
            pitch_vals = pitch.selected_array['frequency']
            pitch_vals = pitch_vals[pitch_vals > 0]
            
            if len(pitch_vals) > 10:
                pitch_range = np.ptp(pitch_vals)
                pitch_std = np.std(pitch_vals)
                intonation = min(1, (pitch_range / 150) * (pitch_std / 30))
            else:
                intonation = 0.4
        except:
            intonation = 0.4
        
        # 3. Intensity variation (word stress)
        try:
            intensity = sound.to_intensity()
            intensity_vals = intensity.values[0]
            intensity_std = np.std(intensity_vals)
            stress = min(1, intensity_std / 15)  # Normalize
        except:
            stress = 0.5
        
        # Pronunciation score (0-9)
        pron_score = (clarity * 0.4 + intonation * 0.3 + stress * 0.3) * 9
        
        # FLUENCY FEATURES
        
        try:
            intensity = sound.to_intensity()
            intensity_vals = intensity.values[0]
            times = intensity.xs()
            
            # Speech detection threshold
            threshold = np.mean(intensity_vals) * 0.4
            
            # Detect speech segments
            speech_segments = intensity_vals > threshold
            
            # Calculate speaking time
            frame_duration = duration / len(intensity_vals)
            speaking_frames = np.sum(speech_segments)
            speaking_time = speaking_frames * frame_duration
            
            # Phonation Time Ratio
            ptr = speaking_time / duration if duration > 0 else 0
            
            # Pause analysis
            silences = []
            in_silence = False
            silence_start = 0
            
            for i, val in enumerate(intensity_vals):
                t = i * frame_duration
                if val < threshold and not in_silence:
                    in_silence = True
                    silence_start = t
                elif val >= threshold and in_silence:
                    in_silence = False
                    if t - silence_start > 0.3:
                        silences.append((silence_start, t))
            
            pause_time = sum(end-start for start, end in silences)
            pause_ratio = pause_time / duration if duration > 0 else 0
            
            # Speech rate (syllable approximation)
            peaks = []  # Syllable proxies
            for i in range(1, len(intensity_vals)-1):
                if (intensity_vals[i] > intensity_vals[i-1] and 
                    intensity_vals[i] > intensity_vals[i+1] and 
                    intensity_vals[i] > threshold*1.5):
                    peaks.append(i)
            
            syllable_count = len(peaks)
            speech_rate = syllable_count / speaking_time if speaking_time > 0 else 0
            
            # Fluency scoring based on IELTS benchmarks
            # Band 9: PTR > 0.8, Speech rate 4-5, Pause ratio < 0.1
            # Band 6: PTR ~0.6, Speech rate 3-4, Pause ratio ~0.2
            # Band 4: PTR < 0.5, Speech rate < 2.5, Pause ratio > 0.3
            
            ptr_score = min(9, ptr * 11.25)  # 0.8 → 9.0
            
            if speech_rate < 2.5:
                rate_score = (speech_rate / 2.5) * 5
            elif speech_rate <= 5.0:
                rate_score = 5 + ((speech_rate - 2.5) / 2.5) * 4
            else:
                rate_score = 9 - min(2, (speech_rate - 5.0) / 1)
            
            pause_score = 9 * (1 - min(1, pause_ratio * 3))
            
            fluency_score = (ptr_score * 0.4 + rate_score * 0.4 + pause_score * 0.2)
            
            details = {
                'praat': {
                    'pronunciation': round(pron_score, 1),
                    'fluency': round(fluency_score, 1),
                    'phonation_ratio': round(ptr, 2),
                    'speech_rate': round(speech_rate, 1),
                    'pause_ratio': round(pause_ratio, 2),
                    'duration': round(duration, 2)
                }
            }
            
            return pron_score, fluency_score, details
            
        except Exception as e:
            print(f"Praat fluency failed: {e}")
            return pron_score, 5.0, {'praat': {'pronunciation': round(pron_score, 1)}}
        
    except Exception as e:
        print(f"Praat analysis failed: {e}")
        # Return default scores
        return 5.0, 5.0, {'praat': {'error': str(e)}}

# SIMPLE TEXT ANALYSIS

def analyze_text_simple(text):
    """
    Simple text analysis for coherence, lexical, grammar
    Returns: (coherence_score, lexical_score, grammar_score, details)
    """
    if not text.strip():
        return 5.0, 5.0, 5.0, {}
    
    words = [w.lower() for w in tokenize(text) if w.isalpha()]
    sentences = sent_tokenize(text)
    
    #  COHERENCE
    linking_words = [
        'because', 'however', 'therefore', 'although', 'furthermore',
        'moreover', 'consequently', 'nevertheless', 'otherwise',
        'for example', 'for instance', 'in addition'
    ]
    
    linking_count = 0
    text_lower = text.lower()
    for word in linking_words:
        linking_count += text_lower.count(word)
    
    linking_density = linking_count / len(sentences) if sentences else 0
    
    # IELTS coherence mapping
    if linking_density > 0.8:
        coherence = 8.0
    elif linking_density > 0.5:
        coherence = 7.0
    elif linking_density > 0.3:
        coherence = 6.0
    elif linking_density > 0.1:
        coherence = 5.0
    else:
        coherence = 4.0
    
    # LEXICAL RESOURCE
    if not words:
        lexical = 5.0
        ttr = 0
    else:
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # IELTS lexical mapping based on TTR
        if ttr > 0.7:
            lexical = 8.0
        elif ttr > 0.6:
            lexical = 7.0
        elif ttr > 0.5:
            lexical = 6.0
        elif ttr > 0.4:
            lexical = 5.0
        elif ttr > 0.3:
            lexical = 4.0
        else:
            lexical = 3.0
        
        # Bonus for longer words (>6 letters)
        long_words = sum(1 for w in words if len(w) > 6)
        if long_words / len(words) > 0.1:
            lexical = min(9.0, lexical + 0.5)
    
    # GRAMMAR
    if not sentences:
        grammar = 5.0
        error_rate = 0
    else:
        # Simple error detection
        text_lower = text.lower()
        
        # Count potential errors
        errors = 0
        error_patterns = [
            (r'\bi is\b', 2),  # Basic agreement
            (r'\bthey is\b', 2),
            (r'\bhe are\b', 2),
            (r'\bshe are\b', 2),
            (r'\bit are\b', 2),
            (r'\ban university\b', 1),  # Article error
            (r'\ban hour\b', -1),  # Actually correct
            (r'\ba hour\b', 1),  # Error
        ]
        
        for pattern, weight in error_patterns:
            errors += len(re.findall(pattern, text_lower)) * weight
        
        # Count complex structures
        complex_markers = ['because', 'although', 'if', 'when', 'which', 'that', 'who']
        complex_count = sum(text_lower.count(marker) for marker in complex_markers)
        
        # Error rate
        error_rate = errors / len(words) if words else 0
        
        # Complexity ratio
        complexity = complex_count / len(sentences) if sentences else 0
        
        # IELTS grammar mapping
        if error_rate < 0.02 and complexity > 0.8:
            grammar = 8.0
        elif error_rate < 0.05 and complexity > 0.5:
            grammar = 7.0
        elif error_rate < 0.1 and complexity > 0.3:
            grammar = 6.0
        elif error_rate < 0.2:
            grammar = 5.0
        elif error_rate < 0.3:
            grammar = 4.0
        else:
            grammar = 3.0
    
    details = {
        'text': {
            'coherence': round(coherence, 1),
            'lexical': round(lexical, 1),
            'grammar': round(grammar, 1),
            'linking_words': linking_count,
            'type_token_ratio': round(ttr, 2),
            'error_rate': round(error_rate, 3),
            'sentence_count': len(sentences),
            'word_count': len(words)
        }
    }
    
    return coherence, lexical, grammar, details

# MAIN BAND CALCULATION

def calculate_band_hybrid(audio_file, text):
    """
    Hybrid band calculation using Praat + simple text analysis
    """
    # Praat analysis for pronunciation and fluency
    pron_score, fluency_score, praat_details = analyze_with_praat(audio_file)
    
    # Text analysis for other criteria
    coherence_score, lexical_score, grammar_score, text_details = analyze_text_simple(text)
    
    # Combine for Fluency & Coherence (weighted average)
    fc_score = (fluency_score * 0.7 + coherence_score * 0.3)
    
    # Calculate overall band (average of all criteria)
    scores = {
        'fluency_coherence': fc_score,
        'lexical_resource': lexical_score,
        'grammatical_accuracy': grammar_score,
        'pronunciation': pron_score
    }
    
    avg_score = sum(scores.values()) / 4
    
    # Round to nearest 0.5 (IELTS band scale)
    final_band = round(avg_score * 2) / 2
    
    # Compile all details
    all_details = {
        'scores': {k: round(v, 1) for k, v in scores.items()},
        'total_band': final_band,
        'praat_details': praat_details.get('praat', {}),
        'text_details': text_details.get('text', {})
    }
    
    return final_band, all_details

def print_hybrid_report(band, details):
    """Print hybrid analysis report"""
    print("\n" + "="*60)
    print("IELTS SPEAKING TEST - HYBRID ANALYSIS")
    print("="*60)
    
    scores = details['scores']
    print("\nBAND SCORES:")
    print("-"*40)
    print(f"Fluency & Coherence:   {scores['fluency_coherence']:.1f}")
    print(f"Lexical Resource:       {scores['lexical_resource']:.1f}")
    print(f"Grammatical Accuracy:   {scores['grammatical_accuracy']:.1f}")
    print(f"Pronunciation:          {scores['pronunciation']:.1f}")
    print("-"*40)
    print(f" OVERALL BAND:        {band:.1f}")
    
    # Band interpretation
    print(f"\nBAND INTERPRETATION:")
    if band >= 8.0:
        print("Band 8-9: Expert user - Fluent, accurate, sophisticated")
    elif band >= 7.0:
        print("Band 7: Good user - Operational command, occasional errors")
    elif band >= 6.0:
        print("Band 6: Competent user - Effective despite inaccuracies")
    elif band >= 5.0:
        print("Band 5: Modest user - Partial command, manages basic topics")
    else:
        print("Band <5: Limited user - Basic communication only")
    
    # Praat details
    praat = details.get('praat_details', {})
    if praat and 'error' not in str(praat):
        print(f"\n PRAAT ANALYSIS:")
        for key, value in praat.items():
            print(f"  {key}: {value}")
    
    # Text details
    text = details.get('text_details', {})
    if text:
        print(f"\nTEXT ANALYSIS:")
        for key, value in text.items():
            if key not in ['coherence', 'lexical', 'grammar']:
                print(f"  {key}: {value}")
    

# For backward compatibility
def calculate_band(fc_raw, coherence_raw, lexical_raw,
                   grammar_acc, grammar_range_val, pronunciation_val):
    """Legacy compatibility function"""
    # Convert to 0-9 scale
    fc_score = min(9, fc_raw * 2 + 4)
    coherence_score = min(9, coherence_raw * 9)
    lexical_score = min(9, lexical_raw * 10)
    grammar_score = min(9, grammar_acc / 10)
    
    avg = (fc_score + coherence_score + lexical_score + grammar_score + pronunciation_val) / 5
    final_band = round(avg * 2) / 2
    
    return final_band, {
        "Fluency": round(fc_score, 2),
        "Coherence": round(coherence_score, 2),
        "Lexical": round(lexical_score, 2),
        "Grammar": round(grammar_score, 2),
        "Pronunciation": pronunciation_val
    }

if __name__ == "__main__":
    print("IELTS Scoring Module is ready...")
    print("Useing calculate_band_hybrid(audio_file, text) for analysis")