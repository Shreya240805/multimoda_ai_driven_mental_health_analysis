# **Voice-Interactive Multimodal Mental Health Assessment System**

A real-time mental health screening platform using **facial expressions**, **vocal tone analysis**, **linguistic sentiment analysis**, and **PHQ-9 questionnaire** responses. Built with Flask, Socket.IO, OpenCV + dlib, and librosa for production-ready deployment.

## **📋 Features**

* **Voice-Interactive PHQ-9 Assessment:** Answer depression screening questions verbally  
* **Real-time Multimodal Analysis:** Simultaneous facial, vocal, and linguistic processing  
* **Live Feedback:** See emotion detection and sentiment analysis as you speak  
* **Secure User Dashboard:** Track assessment history, trends, and average PHQ-9 scores  
* **Personalized Recommendations:** Severity-based mental health guidance  
* **Session Recording:** All analysis data saved with timestamps  
* **Responsive Web Interface:** Works on desktop and mobile browsers  

## ---

**🛠️ Project Structure**

You must organize your folders exactly as shown below for the code to work:

MentalHealthAssessment/
│
├── app.py # Main Flask + SocketIO app
├── models.py # Database models (User/Assessment)
├── config.py # Flask configuration
├── templates/ # HTML templates
│ ├── base.html
│ ├── login.html
│ ├── register.html
│ ├── dashboard.html
│ ├── assessment_voice.html
│ └── results.html
├── static/
│ ├── css/style.css
│ └── js/assessment.js
├── facial_analysis.py # OpenCV + dlib facial emotion
├── vocal_linguistic_analysis.py # librosa + TextBlob
├── cardiovascular_analysis.py # Optional HRV analysis
├── shape_predictor_68_face_landmarks.dat # dlib model (REQUIRED)
│
├── instance/ # SQLite database (auto-created)
└── requirements.txt

text

## ---

**🚀 Installation & Setup**

Follow these steps to set up the environment on **Windows** (commands are similar for Mac/Linux).

### **Step 1: Clone & Create Virtual Environment**

1. Open terminal (Command Prompt/PowerShell) in your project folder.
git clone <your-repo>
cd MentalHealthAssessment

text

2. **Create virtual environment:**
python -m venv venv

text

3. **Activate:**
   * **Windows:** `venv\Scripts\activate`
   * **Mac/Linux:** `source venv/bin/activate`

### **Step 2: Install Dependencies**

Create `requirements.txt` and paste:
Flask==2.3.3
Flask-SocketIO==5.3.6
Flask-Login==0.6.3
Flask-SQLAlchemy==3.0.5
Flask-CORS==4.0.0
eventlet==0.33.3
opencv-python==4.8.1.78
dlib==19.24.2
librosa==0.10.1
speechrecognition==3.10.0
textblob==0.17.1
numpy==1.24.3
scipy==1.11.4
pandas==2.0.3
python-dotenv==1.0.0
werkzeug==2.3.7

text

Install:
pip install -r requirements.txt

text

### **Step 3: Download dlib Face Landmark Model (REQUIRED)**

1. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Extract and place `shape_predictor_68_face_landmarks.dat` in project root

### **Step 4: Configuration**

Create `config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///instance/app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
Create .env:

text
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=sqlite:///instance/app.db
---
🏃‍♂️ How to Run

Single Command Launch
text
python app.py
✅ Database auto-initialized
✅ Server starts at http://127.0.0.1:5000

Usage Flow
Register/Login → Secure dashboard

Start Assessment → Answer 9 PHQ-9 questions verbally

Live Analysis → Facial/vocal/sentiment updates in real-time

View Results → Score + multimodal breakdown + recommendations

---
⚙️ Controls & Customization

Runtime Settings (in app.py)
text
SOCKETIO_PING_INTERVAL = 25    # Real-time heartbeat
SOCKETIO_PING_TIMEOUT = 60     # Connection timeout
Analysis Thresholds (in analysis modules)
text
Facial confidence > 0.5
Vocal MFCC features: 40 coeffs
Sentiment polarity: -1.0 to +1.0
⚠️ Troubleshooting
Issue	Solution
Port 5000 busy	Kill existing Python processes or change port=5001
dlib model not found	Download shape_predictor_68_face_landmarks.dat
float32 JSON error	Update models.py with fixed set_facial_timeline
Camera not working	Check browser permissions, try different webcam
SocketIO disconnects	Check firewall, increase ping_timeout
