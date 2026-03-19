"""
Real-time Vocal and Linguistic Analysis
Processes audio chunks and performs emotion + sentiment analysis
"""
import librosa
import numpy as np
import speech_recognition as sr
from textblob import TextBlob
import joblib
import io
import base64
from scipy.io import wavfile

# Load models
try:
    clf = joblib.load("models/vocal_emotion_model.joblib")
    scaler = joblib.load("models/feature_scaler.joblib")
    MODEL_LOADED = True
    print("✓ Vocal emotion model loaded successfully")
except FileNotFoundError:
    print("⚠ Warning: Vocal emotion model not found. Using neutral predictions.")
    MODEL_LOADED = False


def extract_features_from_audio(audio_data, sample_rate=44100):
    """
    Extract MFCC features from audio data
    
    Args:
        audio_data: Numpy array of audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Scaled feature array
    """
    try:
        # Extract 40 MFCCs
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        
        # Calculate statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        # Combine features (80 total)
        features = np.concatenate((mfcc_mean, mfcc_var))
        
        # Scale if model loaded
        if MODEL_LOADED:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = features.reshape(1, -1)
        
        return features_scaled
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def predict_vocal_emotion(features):
    """
    Predict emotion from audio features
    
    Args:
        features: Scaled feature array
        
    Returns:
        Emotion string
    """
    if not MODEL_LOADED or features is None:
        return "neutral"
    
    try:
        emotion = clf.predict(features)[0]
        return emotion.lower()
    except Exception as e:
        print(f"Vocal prediction error: {e}")
        return "neutral"


def analyze_audio_chunk(audio_blob, sample_rate=44100):
    """
    Analyze audio chunk for vocal emotion
    
    Args:
        audio_blob: Base64 encoded audio or raw bytes
        sample_rate: Sample rate
        
    Returns:
        Emotion string
    """
    try:
        # Decode if base64
        if isinstance(audio_blob, str):
            if ',' in audio_blob:
                audio_blob = audio_blob.split(',')[1]
            audio_bytes = base64.b64decode(audio_blob)
        else:
            audio_bytes = audio_blob
        
        # Convert to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_data = audio_data / 32768.0  # Normalize
        
        # Extract features
        features = extract_features_from_audio(audio_data, sample_rate)
        
        # Predict emotion
        emotion = predict_vocal_emotion(features)
        
        return emotion
    
    except Exception as e:
        print(f"Audio chunk analysis error: {e}")
        return "neutral"


def transcribe_audio(audio_bytes):
    """
    Transcribe audio to text
    
    Args:
        audio_bytes: Raw audio bytes (WAV format)
        
    Returns:
        Transcribed text string
    """
    recognizer = sr.Recognizer()
    
    try:
        # Create audio data from bytes
        audio_data = sr.AudioData(audio_bytes, sample_rate=44100, sample_width=2)
        
        # Transcribe using Google
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Transcription API error: {e}")
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def analyze_sentiment(text):
    """
    Analyze sentiment of text
    
    Args:
        text: String to analyze
        
    Returns:
        Dictionary with polarity and subjectivity
    """
    if not text or len(text.strip()) == 0:
        return {'polarity': 0.2, 'subjectivity': 0.2}
    
    try:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0}


def aggregate_vocal_emotions(vocal_timeline):
    """
    Get dominant vocal emotion from timeline
    
    Args:
        vocal_timeline: List of emotion strings
        
    Returns:
        Most common emotion
    """
    if not vocal_timeline:
        return "neutral"
    
    from collections import Counter
    emotion_counts = Counter(vocal_timeline)
    return emotion_counts.most_common(1)[0][0]


# Test
if __name__ == "__main__":
    print("Vocal & linguistic analysis module loaded")
    print(f"Model status: {'Loaded' if MODEL_LOADED else 'Not loaded'}")
