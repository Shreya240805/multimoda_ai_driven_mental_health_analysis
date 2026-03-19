"""
Mental Health Assessment - Voice-Interactive Multimodal Analysis System
Real-time voice responses with facial, vocal, and linguistic analysis
"""

from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import re
from datetime import datetime
import json
import base64
import time
from collections import Counter

from models import db, User, Assessment
from config import Config
from cardiovascular_analysis import analyze_cardiovascular_stress
from facial_analysis import analyze_base64_frame, aggregate_emotions
from vocal_linguistic_analysis import analyze_audio_chunk, transcribe_audio, analyze_sentiment, aggregate_vocal_emotions

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Initialize extensions
db.init_app(app)
CORS(app)
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='eventlet',
                   ping_timeout=60,
                   ping_interval=25)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'error'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# PHQ-9 Questions (for reference)
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead, or of hurting yourself in some way?"
]

# Store real-time session data (in production, use Redis)
active_sessions = {}

# ============================================
# HELPER FUNCTIONS
# ============================================

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def create_session_data(user_id):
    """Initialize session data for real-time analysis"""
    session_id = f"{user_id}_{int(time.time())}"
    active_sessions[session_id] = {
        'user_id': user_id,
        'start_time': datetime.utcnow(),
        'facial_timeline': [],
        'vocal_timeline': [],
        'audio_chunks': [],
        'transcribed_texts': [],
        'voice_responses': [],
        'has_cardiovascular': False,
        'cv_data': None
    }
    return session_id


def get_session_data(session_id):
    """Retrieve session data"""
    return active_sessions.get(session_id)


def clear_session_data(session_id):
    """Clear session data after assessment"""
    if session_id in active_sessions:
        del active_sessions[session_id]


# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.route('/')
def index():
    """Home page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        errors = []
        
        # Validation
        if len(username) < 3:
            errors.append('Username must be at least 3 characters long')
        if len(username) > 20:
            errors.append('Username must be less than 20 characters')
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            errors.append('Username can only contain letters, numbers, and underscores')
        
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            errors.append('Invalid email address')
        
        if len(password) < 8:
            errors.append('Password must be at least 8 characters long')
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        if User.query.filter_by(username=username).first():
            errors.append('Username already exists')
        
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html')
        
        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False) == 'true'
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))


# ============================================
# DASHBOARD
# ============================================

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with assessment history"""
    assessments = Assessment.query.filter_by(user_id=current_user.id)\
                                   .order_by(Assessment.timestamp.desc())\
                                   .all()
    
    total_assessments = len(assessments)
    
    if total_assessments > 0:
        latest_assessment = assessments[0]
        avg_phq9_score = sum(a.phq9_total for a in assessments) / total_assessments
    else:
        latest_assessment = None
        avg_phq9_score = 0
    
    return render_template('dashboard.html', 
                         assessments=assessments,
                         total_assessments=total_assessments,
                         latest_assessment=latest_assessment,
                         avg_phq9_score=avg_phq9_score)


# ============================================
# VOICE ASSESSMENT ROUTES
# ============================================

@app.route('/assessment', methods=['GET', 'POST'])
@login_required
def assessment():
    """Voice-interactive assessment page"""
    if request.method == 'GET':
        # Create new session for this assessment
        session_id = create_session_data(current_user.id)
        session['assessment_session_id'] = session_id
        
        return render_template('assessment_voice.html',
                             session_id=session_id)
    
    # POST: Handle assessment completion
    return process_voice_assessment_submission()


def process_voice_assessment_submission():
    """Process completed voice assessment and save to database"""
    try:
        session_id = session.get('assessment_session_id')
        if not session_id or session_id not in active_sessions:
            flash('Session expired. Please start a new assessment.', 'error')
            return redirect(url_for('assessment'))
        
        session_data = get_session_data(session_id)
        
        # Get assessment data from form (JSON from voice assessment)
        assessment_json = request.form.get('assessment_data')
        
        if assessment_json:
            voice_data = json.loads(assessment_json)
            
            # Create new assessment record
            new_assessment = Assessment(user_id=current_user.id)
            
            # Calculate PHQ-9 from voice responses
            responses = voice_data.get('responses', [])
            
            # Simplified keyword-based scoring (0-3 scale)
            total_score = 0
            for i, response_data in enumerate(responses[:9]):
                response_text = response_data.get('response', '').lower()
                
                # Simple scoring based on keywords
                score = 0
                if any(word in response_text for word in ['never', 'not at all', 'rarely', 'no', 'none']):
                    score = 0
                elif any(word in response_text for word in ['sometimes', 'occasionally', 'a few', 'once']):
                    score = 1
                elif any(word in response_text for word in ['often', 'frequently', 'many', 'several']):
                    score = 2
                elif any(word in response_text for word in ['always', 'every day', 'constantly', 'all the time', 'daily']):
                    score = 3
                else:
                    # Analyze sentiment for ambiguous responses
                    sentiment = analyze_sentiment(response_text)
                    if sentiment['polarity'] < -0.3:
                        score = 2  # Negative sentiment = higher score
                    elif sentiment['polarity'] > 0.3:
                        score = 0  # Positive sentiment = lower score
                    else:
                        score = 1  # Neutral
                
                if i < 9:  # Only first 9 questions for PHQ-9
                    setattr(new_assessment, f'phq9_q{i+1}', score)
                    total_score += score
            
            new_assessment.phq9_total = total_score
            
            # Process facial emotion data
            if session_data['facial_timeline']:
                facial_results = aggregate_emotions(session_data['facial_timeline'])
                new_assessment.facial_dominant_emotion = facial_results['dominant_emotion']
                new_assessment.facial_emotion_confidence = facial_results['confidence']
                new_assessment.set_facial_timeline(facial_results['timeline'])
                new_assessment.set_emotion_distribution(facial_results['distribution'])
            
            # Process vocal emotion data
            if session_data['vocal_timeline']:
                dominant_vocal = aggregate_vocal_emotions(session_data['vocal_timeline'])
                new_assessment.vocal_dominant_emotion = dominant_vocal
            
            # Process linguistic data
            full_transcript = ' '.join([r.get('response', '') for r in responses])
            if full_transcript:
                new_assessment.transcribed_text = full_transcript
                sentiment = analyze_sentiment(full_transcript)
                new_assessment.sentiment_polarity = sentiment['polarity']
                new_assessment.sentiment_subjectivity = sentiment['subjectivity']
            
            # Calculate session duration
            if voice_data.get('startTime') and voice_data.get('endTime'):
                duration = (voice_data['endTime'] - voice_data['startTime']) / 1000
                new_assessment.session_duration = int(duration)
            
            # Generate recommendations
            severity, recommendations = generate_recommendations(new_assessment, None, session_data)
            new_assessment.severity_level = severity
            new_assessment.recommendations = recommendations
            
            # Save to database
            db.session.add(new_assessment)
            db.session.commit()
            
            # Clear session data
            clear_session_data(session_id)
            session.pop('assessment_session_id', None)
            
            flash('Voice assessment completed successfully!', 'success')
            return redirect(url_for('results', assessment_id=new_assessment.id))
        
        else:
            flash('No assessment data received', 'error')
            return redirect(url_for('assessment'))
    
    except Exception as e:
        db.session.rollback()
        print(f"Assessment submission error: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred processing your assessment.', 'error')
        return redirect(url_for('assessment'))


# ============================================
# REAL-TIME SOCKETIO HANDLERS
# ============================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'success'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")


@socketio.on('join_assessment')
def handle_join_assessment(data):
    """Client joins assessment session"""
    session_id = data.get('session_id')
    if session_id and session_id in active_sessions:
        join_room(session_id)
        emit('joined_session', {'session_id': session_id, 'status': 'success'})
        print(f"Client joined session: {session_id}")
    else:
        emit('error', {'message': 'Invalid session'})


@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame for facial emotion detection"""
    try:
        session_id = data.get('session_id')
        frame_data = data.get('frame')
        timestamp = data.get('timestamp', 0)
        
        if not session_id or session_id not in active_sessions:
            return
        
        session_data = get_session_data(session_id)
        
        # Analyze frame
        result = analyze_base64_frame(frame_data)
        
        if result:
            result['timestamp'] = timestamp
            session_data['facial_timeline'].append(result)
            
            # Send result back to client
            emit('facial_emotion', {
                'emotion': result['dominant_emotion'],
                'confidence': result['confidence'],
                'timestamp': timestamp
            }, room=session_id)
    
    except Exception as e:
        print(f"Video frame error: {e}")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process audio chunk for vocal emotion detection"""
    try:
        session_id = data.get('session_id')
        audio_data = data.get('audio')
        timestamp = data.get('timestamp', 0)
        
        if not session_id or session_id not in active_sessions:
            return
        
        session_data = get_session_data(session_id)
        
        # Analyze audio for vocal emotion
        vocal_emotion = analyze_audio_chunk(audio_data)
        
        if vocal_emotion:
            session_data['vocal_timeline'].append(vocal_emotion)
            session_data['audio_chunks'].append(audio_data)
            
            # Send result back
            emit('vocal_emotion', {
                'emotion': vocal_emotion,
                'timestamp': timestamp
            }, room=session_id)
    
    except Exception as e:
        print(f"Audio chunk error: {e}")


@socketio.on('analyze_text')
def handle_analyze_text(data):
    """Analyze transcribed text for sentiment and keywords"""
    try:
        session_id = data.get('session_id')
        text = data.get('text')
        timestamp = data.get('timestamp', 0)
        
        if not session_id or session_id not in active_sessions:
            return
        
        session_data = get_session_data(session_id)
        
        # Store transcript
        session_data['transcribed_texts'].append(text)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(text)
        
        # Detect positive and negative keywords
        positive_words = ['good', 'happy', 'better', 'enjoy', 'love', 'great', 'excellent', 'wonderful', 'positive', 'excited', 'glad', 'thankful']
        negative_words = ['bad', 'sad', 'worse', 'hate', 'terrible', 'awful', 'depressed', 'hopeless', 'never', 'nothing', 'nobody', 'tired', 'exhausted', 'lonely', 'worthless','stupid','low','tired','despise','alone']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Send result back
        emit('sentiment_result', {
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
            'positive_words': pos_count,
            'negative_words': neg_count,
            'timestamp': timestamp
        }, room=session_id)
    
    except Exception as e:
        print(f"Text analysis error: {e}")


# ============================================
# RESULTS PAGE
# ============================================

@app.route('/results/<int:assessment_id>')
@login_required
def results(assessment_id):
    """Display assessment results"""
    assessment = Assessment.query.get_or_404(assessment_id)
    
    # Security check
    if assessment.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))
    
    # Get emotion distribution
    emotion_dist = assessment.get_emotion_distribution()
    
    return render_template('results.html', 
                         assessment=assessment,
                         emotion_distribution=emotion_dist)


# ============================================
# RECOMMENDATION GENERATION
# ============================================

def generate_recommendations(assessment, cv_results=None, session_data=None):
    """Generate personalized recommendations based on multimodal analysis"""
    phq9_score = assessment.phq9_total
    
    # Determine severity
    if phq9_score <= 4:
        severity = "None-Minimal"
    elif phq9_score <= 9:
        severity = "Mild"
    elif phq9_score <= 14:
        severity = "Moderate"
    elif phq9_score <= 19:
        severity = "Moderately Severe"
    else:
        severity = "Severe"
    
    recommendations = []
    
    # PHQ-9 based recommendations
    if phq9_score <= 4:
        recommendations.append("✓ Your responses suggest minimal or no depression symptoms.")
        recommendations.append("Continue healthy habits: exercise, sleep (7-9 hours), social connections.")
        recommendations.append("Practice mindfulness and stress management techniques.")
        
    elif phq9_score <= 9:
        recommendations.append("Your responses indicate mild depression symptoms.")
        recommendations.append("Lifestyle modifications:")
        recommendations.append("  • Regular physical activity (30 min, 5 days/week)")
        recommendations.append("  • Consistent sleep schedule")
        recommendations.append("  • Balanced diet and hydration")
        recommendations.append("  • Social engagement")
        recommendations.append("Monitor symptoms. Consult professional if they persist beyond 2 weeks.")
        
    elif phq9_score <= 14:
        recommendations.append("⚠️ Your responses suggest moderate depression.")
        recommendations.append("We recommend consulting a mental health professional.")
        recommendations.append("Treatment options:")
        recommendations.append("  • Cognitive Behavioral Therapy (CBT)")
        recommendations.append("  • Counseling or psychotherapy")
        recommendations.append("  • Lifestyle modifications")
        recommendations.append("  • Consider medication evaluation if recommended")
        
    else:
        recommendations.append("⚠️ IMPORTANT: Your responses indicate significant depression.")
        recommendations.append("Please seek professional help immediately:")
        recommendations.append("  • Contact a mental health professional")
        recommendations.append("  • Visit your primary care physician")
       
        
        if assessment.phq9_q9 >= 1:
            recommendations.append("")
            recommendations.append("⚠️ You indicated thoughts of self-harm. This is serious.")
            recommendations.append("Please reach out immediately to someone you trust or call a crisis helpline.")
    
    # Multimodal insights
    if assessment.facial_dominant_emotion:
        recommendations.append("")
        recommendations.append("--- Facial Expression Analysis ---")
        recommendations.append(f"Dominant emotion: {assessment.facial_dominant_emotion.title()}")
        
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        if assessment.facial_dominant_emotion in negative_emotions:
            recommendations.append("Your facial expressions suggest emotional distress.")
    
    if assessment.vocal_dominant_emotion:
        recommendations.append("")
        recommendations.append("--- Voice Analysis ---")
        recommendations.append(f"Vocal emotion: {assessment.vocal_dominant_emotion.title()}")
        
        if assessment.vocal_dominant_emotion in ['sad', 'angry', 'fear']:
            recommendations.append("Your voice patterns indicate negative emotions.")
    
    if assessment.sentiment_polarity is not None:
        recommendations.append("")
        recommendations.append("--- Language Analysis ---")
        
        if assessment.sentiment_polarity < -0.3:
            recommendations.append(f"Your language shows negative sentiment (score: {assessment.sentiment_polarity:.2f})")
            recommendations.append("Consider talking with someone you trust about your feelings.")
        elif assessment.sentiment_polarity > 0.3:
            recommendations.append(f"Your language shows positive sentiment (score: {assessment.sentiment_polarity:.2f})")
    
    # General wellness
    recommendations.append("")
    recommendations.append("--- General Wellness Tips ---")
    recommendations.append("• Deep breathing exercises: 5-10 minutes daily")
    recommendations.append("• Limit caffeine and alcohol")
    recommendations.append("• Spend time in nature/sunlight")
    recommendations.append("• Journal your thoughts")
    recommendations.append("• Set small, achievable daily goals")
    
    return severity, '\n'.join(recommendations)


# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    flash('Page not found', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    db.session.rollback()
    flash('An internal error occurred', 'error')
    return redirect(url_for('index'))


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("=" * 60)
        print("DATABASE INITIALIZED")
        print("=" * 60)
        print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
        print("=" * 60)
    
    print("\n🚀 Starting Mental Health Assessment Server...")
    print("📍 Access at: http://127.0.0.1:5000")
    print("⚡ Real-time voice mode: ENABLED")
    print("🎤 Voice-interactive assessment: READY")
    print("\n")
    
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
