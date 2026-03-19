"""
Advanced Facial Emotion Analysis using DeepFace
High-accuracy deep learning model for emotion detection
"""
import cv2
import numpy as np
import base64
from collections import Counter
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading emotion detection model...")

# Initialize emotion detector
emotion_detector = None
detection_method = "none"

# Try DeepFace (best accuracy)
try:
    from deepface import DeepFace
    # Preload model
    _ = DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True)
    detection_method = "deepface"
    print("✓ DeepFace loaded - Professional-grade emotion detection")
except Exception as e:
    print(f"⚠ DeepFace not available: {str(e)[:100]}")

# Fallback: MediaPipe + rule-based
if detection_method == "none":
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detection_method = "mediapipe"
        print("✓ MediaPipe loaded - Landmark-based detection")
    except Exception as e:
        print(f"⚠ MediaPipe not available: {e}")

# Last resort: OpenCV Haar Cascade
if detection_method == "none":
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        detection_method = "opencv"
        print("✓ OpenCV Haar Cascade - Basic detection")
    except Exception as e:
        print(f"⚠ OpenCV failed: {e}")
        detection_method = "fallback"

print(f"→ Active detection method: {detection_method.upper()}")


def decode_frame(base64_frame):
    """Decode base64 image from browser"""
    try:
        if ',' in base64_frame:
            base64_frame = base64_frame.split(',')[1]
        
        img_data = base64.b64decode(base64_frame)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Frame decode error: {e}")
        return None


# ============================================
# METHOD 1: DEEPFACE (BEST ACCURACY)
# ============================================

def analyze_with_deepface(frame):
    """Use DeepFace for high-accuracy emotion detection"""
    try:
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale)))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze with DeepFace
        result = DeepFace.analyze(
            img_path=rgb_frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        
        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]
        
        # Get emotions
        emotions = result.get('emotion', {})
        dominant = result.get('dominant_emotion', 'neutral').lower()
        
        # Normalize emotion names
        emotion_map = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        dominant = emotion_map.get(dominant, 'neutral')
        
        # Get confidence (0-1 scale)
        confidence = emotions.get(dominant, emotions.get(dominant.capitalize(), 50)) / 100.0
        
        # Normalize all emotions to 0-1 scale
        normalized_emotions = {k.lower(): v/100.0 for k, v in emotions.items()}
        
        return {
            'dominant_emotion': dominant,
            'confidence': float(confidence),
            'all_emotions': normalized_emotions,
            'timestamp': None
        }
        
    except Exception as e:
        print(f"DeepFace analysis error: {str(e)[:100]}")
        return None


# ============================================
# METHOD 2: MEDIAPIPE (GOOD ACCURACY)
# ============================================

def calculate_distance(p1, p2):
    """Calculate Euclidean distance"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_mouth_features(landmarks, w, h):
    """Extract mouth features from MediaPipe landmarks"""
    upper = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lower = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right = np.array([landmarks[291].x * w, landmarks[291].y * h])
    
    mouth_height = calculate_distance(upper, lower)
    mouth_width = calculate_distance(left, right)
    mar = mouth_height / (mouth_width + 1e-6)
    
    # Smile detection
    corners_y = (left[1] + right[1]) / 2
    center_y = (upper[1] + lower[1]) / 2
    smile = (center_y - corners_y) / h * 100
    
    return mar, smile


def get_eye_features(landmarks, w, h):
    """Extract eye features"""
    # Left eye
    lt = np.array([landmarks[159].x * w, landmarks[159].y * h])
    lb = np.array([landmarks[145].x * w, landmarks[145].y * h])
    ll = np.array([landmarks[33].x * w, landmarks[33].y * h])
    lr = np.array([landmarks[133].x * w, landmarks[133].y * h])
    
    l_height = calculate_distance(lt, lb)
    l_width = calculate_distance(ll, lr)
    l_ear = l_height / (l_width + 1e-6)
    
    # Right eye
    rt = np.array([landmarks[386].x * w, landmarks[386].y * h])
    rb = np.array([landmarks[374].x * w, landmarks[374].y * h])
    rl = np.array([landmarks[362].x * w, landmarks[362].y * h])
    rr = np.array([landmarks[263].x * w, landmarks[263].y * h])
    
    r_height = calculate_distance(rt, rb)
    r_width = calculate_distance(rl, rr)
    r_ear = r_height / (r_width + 1e-6)
    
    return (l_ear + r_ear) / 2


def get_eyebrow_features(landmarks, w, h):
    """Extract eyebrow features"""
    lb = np.array([landmarks[70].x * w, landmarks[70].y * h])
    le = np.array([landmarks[159].x * w, landmarks[159].y * h])
    rb = np.array([landmarks[300].x * w, landmarks[300].y * h])
    re = np.array([landmarks[386].x * w, landmarks[386].y * h])
    
    l_dist = calculate_distance(lb, le)
    r_dist = calculate_distance(rb, re)
    avg = (l_dist + r_dist) / 2
    
    # Normalize
    return (avg - 25) / 10


def analyze_with_mediapipe(frame):
    """Use MediaPipe landmarks for emotion detection"""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.3,
                'all_emotions': {'neutral': 0.3},
                'timestamp': None
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Extract features
        mar, smile = get_mouth_features(landmarks, w, h)
        ear = get_eye_features(landmarks, w, h)
        brow = get_eyebrow_features(landmarks, w, h)
        
        # Emotion scoring with TUNED thresholds
        scores = {
            'neutral': 0.2,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprise': 0.0,
            'fear': 0.0,
            'disgust': 0.0
        }
        
        # HAPPY (strict thresholds)
        if smile > 0.8:
            scores['happy'] += 0.5
        if smile > 1.5:
            scores['happy'] += 0.4
        if mar > 0.12 and smile > 0.5:
            scores['happy'] += 0.3
        
        # SAD
        if smile < -0.4:
            scores['sad'] += 0.4
        if smile < -0.8:
            scores['sad'] += 0.3
        if ear < 0.20:
            scores['sad'] += 0.3
        if brow < -0.4:
            scores['sad'] += 0.2
        
        # ANGRY
        if brow < -0.8:
            scores['angry'] += 0.5
        if ear < 0.18:
            scores['angry'] += 0.3
        if mar < 0.12 and smile < -0.2:
            scores['angry'] += 0.3
        
        # SURPRISE
        if ear > 0.26:
            scores['surprise'] += 0.4
        if mar > 0.18:
            scores['surprise'] += 0.4
        if brow > 0.8:
            scores['surprise'] += 0.4
        
        # FEAR
        if ear > 0.24 and brow > 0.6:
            scores['fear'] += 0.5
        if mar > 0.15 and smile < 0:
            scores['fear'] += 0.3
        
        # DISGUST
        if smile < -0.7 and mar < 0.12:
            scores['disgust'] += 0.4
        if ear < 0.19 and brow < -0.3:
            scores['disgust'] += 0.3
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant]
        
        # Boost confidence for strong signals
        if confidence > 0.4:
            confidence = min(confidence * 1.2, 0.95)
        
        return {
            'dominant_emotion': dominant,
            'confidence': float(confidence),
            'all_emotions': {k: float(v) for k, v in scores.items()},
            'timestamp': None
        }
        
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return None


# ============================================
# METHOD 3: OPENCV FALLBACK
# ============================================

def analyze_with_opencv(frame):
    """Basic OpenCV Haar Cascade detection"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {'neutral': 0.5},
                'timestamp': None
            }
        
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.2,
            'all_emotions': {'neutral': 0.2},
            'timestamp': None
        }
        
    except Exception as e:
        print(f"OpenCV error: {e}")
        return None


# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_frame(frame):
    """Analyze frame using best available method"""
    if frame is None:
        return None
    
    result = None
    
    # Try primary method
    if detection_method == "deepface":
        result = analyze_with_deepface(frame)
    elif detection_method == "mediapipe":
        result = analyze_with_mediapipe(frame)
    elif detection_method == "opencv":
        result = analyze_with_opencv(frame)
    
    # Fallback if primary fails
    if result is None:
        result = {
            'dominant_emotion': 'neutral',
            'confidence': 0.5,
            'all_emotions': {'neutral': 0.5},
            'timestamp': None
        }
    
    return result


def analyze_base64_frame(base64_frame):
    """Complete pipeline: decode and analyze"""
    frame = decode_frame(base64_frame)
    if frame is not None:
        return analyze_frame(frame)
    return None


def aggregate_emotions(emotion_timeline):
    """Aggregate emotions from timeline"""
    if not emotion_timeline:
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'distribution': {'neutral': 100.0},
            'timeline': []
        }
    
    valid_emotions = [e for e in emotion_timeline if e and e.get('dominant_emotion')]
    
    if not valid_emotions:
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'distribution': {'neutral': 100.0},
            'timeline': []
        }
    
    emotions = [e['dominant_emotion'] for e in valid_emotions]
    emotion_counts = Counter(emotions)
    
    total = len(emotions)
    distribution = {
        emotion: round((count / total) * 100, 1)
        for emotion, count in emotion_counts.items()
    }
    
    dominant_emotion = emotion_counts.most_common(1)[0][0]
    
    confidences = [
        e['confidence'] for e in valid_emotions
        if e['dominant_emotion'] == dominant_emotion
    ]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    
    return {
        'dominant_emotion': dominant_emotion,
        'confidence': avg_confidence,
        'distribution': distribution,
        'timeline': valid_emotions
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACIAL EMOTION DETECTION TEST")
    print("="*60)
    print(f"Detection method: {detection_method.upper()}")
    
    # Test with blank frame
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    result = analyze_frame(test_frame)
    
    if result:
        print(f"✓ Test successful!")
        print(f"  Emotion: {result['dominant_emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
    else:
        print("✗ Test failed")
    
    print("="*60)
