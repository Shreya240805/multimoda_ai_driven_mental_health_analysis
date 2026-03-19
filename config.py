"""
Application Configuration
"""
import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base configuration"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production-2025-mental-health'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'mental_health.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # Upload folders
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    RECORDINGS_FOLDER = os.path.join(basedir, 'recordings')
    MODELS_FOLDER = os.path.join(basedir, 'models')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'webm', 'ogg'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'webm', 'avi'}
    ALLOWED_CV_EXTENSIONS = {'csv'}
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Flask-SocketIO
    SOCKETIO_ASYNC_MODE = 'eventlet'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"  # Restrict in production
    
    # Real-time settings
    VIDEO_FPS = 2  # Process 2 frames per second
    AUDIO_CHUNK_DURATION = 2  # Process audio every 2 seconds
    
    @staticmethod
    def init_app(app):
        """Initialize application directories"""
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RECORDINGS_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
