"""
Database Models for Mental Health Assessment System
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User account with secure authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    assessments = db.relationship(
        'Assessment',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )
    
    def set_password(self, password):
        """Hash password using PBKDF2-SHA256"""
        self.password_hash = generate_password_hash(
            password,
            method='pbkdf2:sha256',
            salt_length=16
        )
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Assessment(db.Model):
    """Mental health assessment with multimodal analysis"""
    __tablename__ = 'assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # PHQ-9 Questionnaire (0-3 scale, 9 questions)
    phq9_q1 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q2 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q3 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q4 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q5 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q6 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q7 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q8 = db.Column(db.Integer, nullable=False, default=0)
    phq9_q9 = db.Column(db.Integer, nullable=False, default=0)
    phq9_total = db.Column(db.Integer, nullable=False, default=0)
    
    # Facial Analysis (Real-time)
    facial_emotions_timeline = db.Column(db.Text)  # JSON array of emotions over time
    facial_dominant_emotion = db.Column(db.String(50))
    facial_emotion_confidence = db.Column(db.Float)
    facial_emotion_distribution = db.Column(db.Text)  # JSON of emotion percentages
    
    # Vocal Analysis (Real-time)
    vocal_emotions_timeline = db.Column(db.Text)  # JSON array
    vocal_dominant_emotion = db.Column(db.String(50))
    
    # Linguistic Analysis (Real-time transcription)
    transcribed_text = db.Column(db.Text)
    sentiment_polarity = db.Column(db.Float)   # -1 to +1
    sentiment_subjectivity = db.Column(db.Float)  # 0 to 1
    
    # Cardiovascular (Optional)
    has_cardiovascular = db.Column(db.Boolean, default=False)
    cv_mean_hr = db.Column(db.Float)
    cv_max_hr = db.Column(db.Float)
    cv_min_hr = db.Column(db.Float)
    cv_stress_score = db.Column(db.Integer)
    cv_stress_level = db.Column(db.String(20))
    cv_hrv_sdnn = db.Column(db.Float)
    cv_hrv_rmssd = db.Column(db.Float)
    
    # Final Assessment
    severity_level = db.Column(db.String(50))
    recommendations = db.Column(db.Text)
    
    # Session data
    session_duration = db.Column(db.Integer)  # seconds

    # ---------- FIXED METHOD ----------
    def set_facial_timeline(self, timeline_list):
        """
        Store facial emotions timeline as JSON.
        Uses json.dumps default to convert non‑serializable values (e.g. float32)
        to normal Python floats.
        """
        self.facial_emotions_timeline = json.dumps(
            timeline_list,
            default=lambda o: float(o)
        )
    # ----------------------------------

    def get_facial_timeline(self):
        """Retrieve facial emotions timeline"""
        if self.facial_emotions_timeline:
            return json.loads(self.facial_emotions_timeline)
        return []
    
    def set_emotion_distribution(self, distribution_dict):
        """Store emotion distribution as JSON"""
        self.facial_emotion_distribution = json.dumps(distribution_dict)
    
    def get_emotion_distribution(self):
        """Retrieve emotion distribution"""
        if self.facial_emotion_distribution:
            return json.loads(self.facial_emotion_distribution)
        return {}
    
    def __repr__(self):
        return f'<Assessment {self.id} - {self.severity_level}>'
