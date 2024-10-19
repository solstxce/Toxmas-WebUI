import logging
import os
import tempfile
from datetime import datetime, timedelta
from functools import wraps
import cv2
import jwt
import numpy as np
import torch
from bson.objectid import ObjectId
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_pymongo import PyMongo
from flasgger import Swagger, swag_from
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip
from nudenet import NudeDetector
from PIL import Image
from pymongo.errors import ConnectionFailure
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.graphics.charts.piecharts import Pie
from transformers import RobertaTokenizer, RobertaModel
from werkzeug.security import generate_password_hash, check_password_hash
from docx import Document
import moviepy.config as cfg
import logging
import os
import tempfile
from datetime import datetime, timedelta
from functools import wraps
import cv2
import jwt
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bson.objectid import ObjectId
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_pymongo import PyMongo
from flasgger import Swagger, swag_from
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip
from nudenet import NudeDetector
from PIL import Image
from pymongo.errors import ConnectionFailure
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.graphics.charts.piecharts import Pie
from reportlab.pdfgen import canvas
from transformers import RobertaTokenizer, RobertaModel
from werkzeug.security import generate_password_hash, check_password_hash
from docx import Document
import moviepy.config as cfg
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords
import whisper
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
cfg.change_settings({"IMAGEMAGICK_BINARY": "magick.exe"})
from better_profanity import profanity
from nltk.corpus import wordnet
import re

# Initialize profanity filter
profanity.load_censor_words()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app, supports_credentials=True)
Swagger(app)

# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/document_analyzer'
mongo = PyMongo(app)

# RoBERTa model setup
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

logger.info("Initializing RoBERTa model...")
model = RobertaClass()
tokenizer = RobertaTokenizer.from_pretrained(r'./tokenizer')
try:
    model.load_state_dict(torch.load("pytorch_roberta_cyberbullying.bin", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    logger.info("RoBERTa model loaded successfully")
except Exception as e:
    logger.error(f"Error loading RoBERTa model: {str(e)}")

# NudeNet Setup
logger.info("Initializing NudeDetector...")
nude_detector = NudeDetector()
logger.info("NudeDetector initialized")

# Verify MongoDB connection
try:
    mongo.db.command('ismaster')    
    logger.info("MongoDB connection successful")
except ConnectionFailure:
    logger.error("MongoDB server not available")

# Helper functions
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            logger.warning("No token provided in request")
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.users.find_one({'_id': ObjectId(data['user_id'])})
            logger.info(f"User authenticated: {current_user['email']}")
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated



# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# RoBERTa model setup

# List of abusive words
abusive_word_list = [
    "fuck", "shit", "asshole", "bitch", "cunt", "damn", "hell", "bastard", "motherfucker",
    "dick", "pussy", "slut", "whore", "idiot", "stupid", "dumb", "retard", "loser", "jerk",
    "moron", "imbecile", "cretin", "twat", "wanker", "prick", "dickhead", "arsehole", "fag",
    "faggot", "homo", "queer", "dyke", "lesbo", "tranny", "nigger", "nigga", "spic", "kike",
    "chink", "gook", "wetback", "beaner", "gringo", "cracker", "redneck", "hillbilly",
    "white trash", "gay", "lgbt", "lesbians", "sexy", "fucking", "porn", "perv", "fart",
    "ass", "boobs", "trash", "booty", "baddie"
]


def predict(text, model, tokenizer, max_len=256):
    model.eval()
    
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    
    ids = inputs['input_ids'].to('cpu', dtype=torch.long)
    mask = inputs['attention_mask'].to('cpu', dtype=torch.long)

    with torch.no_grad():
        outputs = model(ids, mask)
    
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    predicted_probability = probabilities[0][predicted_class].item()
    
    return predicted_class, predicted_probability

def categorize_document(abusive_count, total_words):
    percentage = (abusive_count / total_words) * 100
    if percentage < 2:
        return "Good"
    elif 2 <= percentage <= 5:
        return "Average"
    else:
        return "Bad"

def preprocess_text(text):
    words = text.lower().split()
    return [word for word in words if word not in stop_words]

def create_pie_chart(abusive_count, total_count):
    labels = 'Abusive Words', 'Other Words'
    sizes = [abusive_count, total_count - abusive_count]
    colors = ['red', 'green']
    plt.figure(figsize=(6, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Word Distribution')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        plt.savefig(tmpfile.name, format='png')
        plt.close()
        return tmpfile.name

def analyze_text(text):
    logger.info("Starting text analysis")
    words = preprocess_text(text)
    total_words = len(words)
    abusive_words = []
    abusive_word_counts = Counter()
    
    for word in words:
        if word.lower() in abusive_word_list:
            _, probability = predict(word, model, tokenizer)
            abusive_words.append((word, probability))
            abusive_word_counts[word.lower()] += 1
    
    # Remove duplicates and add count
    abusive_words = [(word, prob, abusive_word_counts[word.lower()]) for word, prob in set(abusive_words)]
    
    document_state = categorize_document(len(set(word.lower() for word, _, _ in abusive_words)), total_words)
    pie_chart_path = create_pie_chart(len(set(word.lower() for word, _, _ in abusive_words)), total_words)
    
    logger.info(f"Text analysis complete. Document state: {document_state}")
    return document_state, abusive_words, total_words, pie_chart_path

def generate_report(document_state, abusive_words, total_words, pie_chart_path, content):
    logger.info("Generating analysis report")
    report_path = tempfile.mktemp(suffix='.pdf')
    
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    width, height = letter

    can.drawString(100, height - 50, "Document Analysis")
    can.drawImage(pie_chart_path, 100, height - 300, width=300, height=200)
    can.drawString(100, height - 350, f"Document State: {document_state}")
    can.drawString(100, height - 370, f"Total words (after removing stopwords): {total_words}")
    can.drawString(100, height - 390, f"Number of unique abusive words found: {len(abusive_words)}")
    can.drawString(100, height - 410, f"Percentage of abusive words: {(len(abusive_words) / total_words) * 100:.2f}%")
    
    y = height - 450
    can.drawString(100, y, "Abusive words, their probabilities, and repetition count:")
    y -= 20
    for word, prob, count in abusive_words:
        can.drawString(120, y, f"{word}: {prob:.4f} (repetition count: {count})")
        y -= 20
        if y < 50:
            can.showPage()
            y = height - 50

    can.showPage()

    words = content.split()
    x, y = 100, height - 50
    for word in words:
        if x + can.stringWidth(word) > width - 100:
            y -= 20
            x = 100
        if y < 50:
            can.showPage()
            y = height - 50
            x = 100
        
        can.saveState()
        if word.lower() in [w.lower() for w, _, _ in abusive_words]:
            can.setFont("Helvetica-Bold", 12)
            can.setFillColor(colors.red)
            can.rect(x, y - 2, can.stringWidth(word), 14, fill=1)
            can.setFillColor(colors.white)
            can.drawString(x, y, word)
        else:
            can.setFont("Helvetica", 12)
            can.setFillColor(colors.black)
            can.drawString(x, y, word)
        can.restoreState()
        
        x += can.stringWidth(word + " ")

    can.save()

    packet.seek(0)
    new_pdf = PdfReader(packet)
    output = PdfWriter()

    for page in new_pdf.pages:
        output.add_page(page)

    with open(report_path, "wb") as output_stream:
        output.write(output_stream)
    
    os.unlink(pie_chart_path)
    
    logger.info(f"Report generated successfully: {report_path}")
    return report_path

def extract_text(file):
    logger.info(f"Extracting text from file: {file.filename}")
    try:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() == '.pdf':
            logger.debug("Extracting text from PDF")
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        elif ext.lower() in ['.doc', '.docx']:
            logger.debug("Extracting text from Word document")
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext.lower() == '.txt':
            logger.debug("Extracting text from TXT file")
            text = file.read().decode('utf-8')
        else:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError("Unsupported file format")
        logger.info(f"Text extraction complete. Extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise

# Expand the list of harmful words
harmful_words = {
    "stupid", "bastard", "idiot", "dumb", "fool", "nasty", "fucking", "shit", "porn", "sex",
    "pornography", "fuck", "bitch", "asshole", "cunt", "dick", "pussy", "whore", "slut",
    "retard", "faggot", "nigger", "spic", "kike", "chink", "gook", "wetback", "beaner",
    "gringo", "cracker", "redneck", "hillbilly", "white trash", "queer", "dyke", "fag",
    "homo", "tranny", "perv", "pedo", "rape", "molest", "kill", "murder", "suicide",
    "terrorist", "bomb", "nazi", "fascist", "racist", "sexist", "homophobe", "transphobe",
    "ableist", "bigot", "hate", "violence", "abuse", "harass", "bully", "threaten"
}

import subprocess
import glob
import shutil

import subprocess
import glob
import shutil
import json

class VideoProcessor:
    def __init__(self):
        self.detector = NudeDetector()
        self.whisper_model = whisper.load_model("base")

    def process_video(self, input_path):
        logger.info(f"Processing video: {input_path}")
        
        try:
            # Create temporary directories for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_dir = os.path.join(temp_dir, 'frames')
                censored_frames_dir = os.path.join(temp_dir, 'censored_frames')
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(censored_frames_dir, exist_ok=True)

                # Get video information
                video_info = self._get_video_info(input_path)
                frame_rate = video_info['frame_rate']
                duration = video_info['duration']
                total_frames = int(frame_rate * duration)

                # Extract frames using FFmpeg
                self._extract_frames(input_path, frames_dir, frame_rate)

                # Process and censor frames
                self._process_frames(frames_dir, censored_frames_dir, total_frames)

                # Reconstruct video from censored frames
                out_path = self._get_output_path(input_path, 'censored_video.mp4')
                self._reconstruct_video(censored_frames_dir, out_path, input_path, frame_rate)

                # Add watermark
                final_output_path = self._add_watermark(out_path, input_path)

            logger.info(f"Video processing complete. Output saved to: {final_output_path}")
            return final_output_path
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise

    def _get_video_info(self, input_path):
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        frame_rate = eval(video_stream['r_frame_rate'])
        duration = float(data['format']['duration'])
        
        return {'frame_rate': frame_rate, 'duration': duration}

    def _extract_frames(self, input_path, frames_dir, frame_rate):
        logger.info("Extracting frames using FFmpeg")
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'fps={frame_rate}',
            os.path.join(frames_dir, 'frame_%06d.png')
        ]
        subprocess.run(cmd, check=True)

    def _process_frames(self, frames_dir, censored_frames_dir, total_frames):
        logger.info("Processing and censoring frames")
        frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        for i, frame_path in enumerate(frame_files):
            try:
                censored_img_path = self.detector.censor(frame_path)
                if censored_img_path:
                    frame_name = os.path.basename(frame_path)
                    censored_frame_path = os.path.join(censored_frames_dir, frame_name)
                    os.rename(censored_img_path, censored_frame_path)
                else:
                    logger.warning(f"Censorship returned None for {frame_path}")
                    # If censorship fails, copy the original frame
                    shutil.copy(frame_path, censored_frames_dir)
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {str(e)}")
                # If processing fails, copy the original frame
                shutil.copy(frame_path, censored_frames_dir)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total_frames} frames")

    def _reconstruct_video(self, censored_frames_dir, out_path, input_path, frame_rate):
        logger.info("Reconstructing video from censored frames")
        cmd = [
            'ffmpeg',
            '-framerate', str(frame_rate),
            '-i', os.path.join(censored_frames_dir, 'frame_%06d.png'),
            '-i', input_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            out_path
        ]
        subprocess.run(cmd, check=True)

    def _add_watermark(self, video_path, input_path):
        logger.info("Adding watermark to video")
        video = VideoFileClip(video_path)
        watermark = (TextClip("Verified by NoHate", fontsize=24, color='white', font='Arial')
                     .set_position(('right', 'bottom'))
                     .set_duration(video.duration))
        
        final_video = CompositeVideoClip([video, watermark])
        final_output_path = self._get_output_path(input_path, 'final_censored_video.mp4')
        final_video.write_videofile(final_output_path, codec='libx264', audio_codec='aac')
        return final_output_path

    def _get_output_path(self, input_path, filename):
        return os.path.join(os.path.dirname(input_path), filename)

    def analyze_image(self, image_path):
        logger.info(f"Starting image analysis for: {image_path}")
        
        try:
            censored_img_path = self.detector.censor(image_path)
            if censored_img_path:
                censored_img = Image.open(censored_img_path)
                original_img = Image.open(image_path)
                
                diff = np.array(original_img) - np.array(censored_img)
                overall_score = np.sum(np.abs(diff)) / (original_img.size[0] * original_img.size[1] * 3)
                
                stats = {
                    'overall_score': overall_score,
                    'censored_image_path': censored_img_path
                }
                
                logger.info(f"Image analysis complete. Overall score: {overall_score}")
                logger.debug(f"Detailed stats: {stats}")
                
                return stats
            else:
                logger.warning(f"Censorship returned None for {image_path}")
                return None
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return None

    def extract_audio(self, video_path):
        video = VideoFileClip(video_path)
        audio_path = tempfile.mktemp(suffix='.wav')
        video.audio.write_audiofile(audio_path)
        video.close()
        return audio_path

    def transcribe_audio_whisper(self, audio_path):
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def convert_audio_to_text_with_timestamps_whisper(self, audio_path, transcribed_text):
        audio = AudioSegment.from_wav(audio_path)
        chunk_length_ms = 2000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        words = transcribed_text.split()
        chunk_size = len(words) // len(chunks) if len(chunks) > 0 else len(words)
        transcription = []
        for i, chunk in enumerate(chunks):
            start_time = i * 2
            chunk_text = " ".join(words[i * chunk_size:(i + 1) * chunk_size])
            transcription.append((start_time, chunk_text))
        return transcription

    def find_harmful_word_timestamps(self, transcript_lines, timestamps):
        mute_times = []
        for i, line in enumerate(transcript_lines):
            words = line.split()
            for word in words:
                if word.lower() in harmful_words:
                    mute_times.append(timestamps[i])
                    break
        return mute_times

    def mute_video_at_timestamps(self, video_file, mute_times):
        video = VideoFileClip(video_file)
        clips = []
        last_end = 0
        for timestamp in mute_times:
            start_time = timestamp
            end_time = start_time + 1
            if start_time > last_end:
                clips.append(video.subclip(last_end, start_time))
            muted_clip = video.subclip(start_time, end_time).volumex(0)
            clips.append(muted_clip)
            last_end = end_time
        if last_end < video.duration:
            clips.append(video.subclip(last_end, video.duration))
        final_clip = concatenate_videoclips(clips)
        output_path = tempfile.mktemp(suffix='.mp4')
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        return output_path

# Add these functions from document_rephrase.py
def get_legal_alternatives(word):
    alternatives = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and not profanity.contains_profanity(lemma.name()):
                alternatives.append(lemma.name().replace('_', ' '))
    return alternatives if alternatives else [word]

def replace_profanity_in_sentence(sentence):
    words = sentence.split()
    replaced_sentence = []
    for word in words:
        cleaned_word = re.sub(r'[^\w\s]', '', word)
        if profanity.contains_profanity(cleaned_word):
            alternatives = get_legal_alternatives(cleaned_word)
            replaced_sentence.append(alternatives[0])
        else:
            replaced_sentence.append(word)
    return ' '.join(replaced_sentence)

def replace_profanity_in_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    replaced_sentences = [replace_profanity_in_sentence(sentence) for sentence in sentences]
    return ' '.join(replaced_sentences)

# Routes
@app.route('/register', methods=['POST'])
@swag_from({
    'tags': ['Authentication'],
    'description': 'Register a new user',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string'},
                    'password': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        '201': {
            'description': 'User created successfully'
        },
        '400': {
            'description': 'User already exists'
        }
    }
})
def register():
    data = request.get_json()
    logger.info(f"Attempting to register user: {data['email']}")
    hashed_password = generate_password_hash(data['password'])
    new_user = {
        'email': data['email'],
        'password': hashed_password
    }
    if mongo.db.users.find_one({'email': data['email']}):
        logger.warning(f"Registration failed: User {data['email']} already exists")
        return jsonify({'message': 'User already exists'}), 400
    mongo.db.users.insert_one(new_user)
    logger.info(f"User registered successfully: {data['email']}")
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/login', methods=['POST'])
@swag_from({
    'tags': ['Authentication'],
    'description': 'Login and receive a JWT token',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string'},
                    'password': {'type': 'string'}
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Login successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'token': {'type': 'string'},
                    'message': {'type': 'string'}
                }
            }
        },
        '401': {
            'description': 'Invalid credentials'
        }
    }
})
def login():
    data = request.get_json()
    logger.info(f"Login attempt for user: {data['email']}")
    user = mongo.db.users.find_one({'email': data['email']})
    if user and check_password_hash(user['password'], data['password']):
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        logger.info(f"Login successful for user: {data['email']}")
        
        # Create the response
        response = jsonify({'token': token, 'message': 'Login successful'})
        
        # Set the token as a cookie
        response.set_cookie('token', token, httponly=True, secure=True, samesite='Strict', max_age=86400)  # max_age is in seconds (24 hours)
        
        return response, 200
    
    logger.warning(f"Login failed for user: {data['email']}")
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/analyze', methods=['POST'])
@token_required
@swag_from({
    'tags': ['Document Analysis'],
    'description': 'Analyze a document, image, or video for inappropriate content',
    'parameters': [
        {
            'name': 'Authorization',
            'in': 'header',
            'type': 'string',
            'required': 'true',
            'description': 'JWT token'
        },
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': 'true',
            'description': 'Document, image, or video to analyze'
        }
    ],
    'responses': {
        '200': {
            'description': 'Analysis report, image stats, or censored video',
            'content': {
                'application/pdf': {},
                'application/json': {},
                'video/mp4': {}
            }
        },
        '400': {
            'description': 'Bad request'
        },
        '401': {
            'description': 'Unauthorized'
        },
        '500': {
            'description': 'Internal server error'
        }
    }
})
def analyze_document(current_user):
    logger.info(f"Analysis request received from user: {current_user['email']}")
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected for uploading")
        return jsonify({'message': 'No file selected for uploading'}), 400
    
    logger.info(f"Processing file: {file.filename}")
    try:
        _, ext = os.path.splitext(file.filename)
        if ext.lower() in ['.pdf', '.doc', '.docx', '.txt']:
            logger.info("Extracting text from document")
            text = extract_text(file)
            
            action = request.form.get('action', 'censor')
            
            if action == 'rephrase':
                logger.info("Rephrasing text")
                processed_text = replace_profanity_in_text(text)
                report_path = generate_rephrased_report(processed_text, text)
            else:  # Default to censoring
                logger.info("Analyzing and censoring text")
                document_state, abusive_words, total_words, pie_chart_path = analyze_text(text)
                report_path = generate_report(document_state, abusive_words, total_words, pie_chart_path, text)
            
            # Store analysis result in MongoDB
            analysis_result = {
                'user_id': current_user['_id'],
                'original_filename': file.filename,
                'result_filename': 'analysis_report.pdf',
                'result_path': report_path,
                'analysis_type': 'document',
                'action': action,
                'created_at': datetime.utcnow()
            }
            mongo.db.analyses.insert_one(analysis_result)
            
            return send_file(report_path, as_attachment=True, download_name='analysis_report.pdf', mimetype='application/pdf')
        elif ext.lower() in ['.jpg', '.jpeg', '.png']:
            logger.info("Processing image file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                logger.info("Analyzing image")
                video_processor = VideoProcessor()
                image_stats = video_processor.analyze_image(temp_file.name)
            os.unlink(temp_file.name)
            
            # Store analysis result in MongoDB
            analysis_result = {
                'user_id': current_user['_id'],
                'original_filename': file.filename,
                'result_filename': 'image_analysis.json',
                'result_data': image_stats,
                'analysis_type': 'image',
                'created_at': datetime.utcnow()
            }
            mongo.db.analyses.insert_one(analysis_result)
            
            return jsonify(image_stats), 200
        elif ext.lower() in ['.mp4', '.avi', '.mov']:
            logger.info("Processing video file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                logger.info("Analyzing and censoring video")
                video_processor = VideoProcessor()
                censored_video_path = video_processor.process_video(temp_file.name)
            
            # Store analysis result in MongoDB
            analysis_result = {
                'user_id': current_user['_id'],
                'original_filename': file.filename,
                'result_filename': 'censored_video.mp4',
                'result_path': censored_video_path,
                'analysis_type': 'video',
                'created_at': datetime.utcnow()
            }
            mongo.db.analyses.insert_one(analysis_result)
            
            return send_file(censored_video_path, as_attachment=True, download_name='censored_video.mp4', mimetype='video/mp4')
        else:
            logger.error(f"Unsupported file format: {ext}")
            return jsonify({'message': 'Unsupported file format'}), 400
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'message': 'An error occurred during analysis'}), 500

@app.route('/user_info', methods=['GET'])
@token_required
@swag_from({
    'tags': ['User'],
    'description': 'Get user information',
    'parameters': [
        {
            'name': 'Authorization',
            'in': 'header',
            'type': 'string',
            'required': 'true',
            'description': 'JWT token'
        }
    ],
    'responses': {
        '200': {
            'description': 'User information',
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string'},
                    'id': {'type': 'string'}
                }
            }
        },
        '401': {
            'description': 'Unauthorized'
        }
    }
})
def get_user_info(current_user):
    logger.info(f"User info request received for user: {current_user['email']}")
    user_info = {
        'email': current_user['email'],
        'id': str(current_user['_id'])
    }
    return jsonify(user_info), 200

# Add a new route to fetch past analyses
@app.route('/past_analyses', methods=['GET'])
@token_required
@swag_from({
    'tags': ['Analysis History'],
    'description': 'Get a list of past analyses for the current user',
    'parameters': [
        {
            'name': 'Authorization',
            'in': 'header',
            'type': 'string',
            'required': 'true',
            'description': 'JWT token'
        }
    ],
    'responses': {
        '200': {
            'description': 'List of past analyses',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'original_filename': {'type': 'string'},
                        'result_filename': {'type': 'string'},
                        'analysis_type': {'type': 'string'},
                        'created_at': {'type': 'string', 'format': 'date-time'}
                    }
                }
            }
        },
        '401': {
            'description': 'Unauthorized'
        }
    }
})
def get_past_analyses(current_user):
    analyses = mongo.db.analyses.find({'user_id': current_user['_id']}).sort('created_at', -1)
    result = []
    for analysis in analyses:
        result.append({
            'id': str(analysis['_id']),
            'original_filename': analysis['original_filename'],
            'result_filename': analysis['result_filename'],
            'analysis_type': analysis['analysis_type'],
            'created_at': analysis['created_at'].isoformat()
        })
    return jsonify(result), 200

# Add a new route to download analysis results
@app.route('/download_analysis/<analysis_id>', methods=['GET'])
@token_required
def download_analysis(current_user, analysis_id):
    analysis = mongo.db.analyses.find_one({'_id': ObjectId(analysis_id), 'user_id': current_user['_id']})
    if not analysis:
        return jsonify({'message': 'Analysis not found'}), 404

    if analysis['analysis_type'] == 'image':
        return jsonify(analysis['result_data']), 200
    else:
        return send_file(analysis['result_path'], as_attachment=True, download_name=analysis['result_filename'])

# Add a new function to generate a report for rephrased text
def generate_rephrased_report(rephrased_text, original_text):
    # Implementation similar to generate_report, but tailored for rephrased text
    # You'll need to create a PDF that shows the original text and the rephrased text side by side
    # Use reportlab to create the PDF
    # Return the path to the generated PDF
    pass

# Add these new routes after the existing routes

@app.route('/recent_analyses', methods=['GET'])
@token_required
def get_recent_analyses(current_user):
    analyses = mongo.db.analyses.find({'user_id': current_user['_id']}).sort('created_at', -1).limit(5)
    result = []
    for analysis in analyses:
        result.append({
            'id': str(analysis['_id']),
            'original_filename': analysis['original_filename'],
            'analysis_type': analysis['analysis_type'],
            'created_at': analysis['created_at'].isoformat()
        })
    return jsonify(result), 200

@app.route('/content_types', methods=['GET'])
@token_required
def get_content_types(current_user):
    pipeline = [
        {'$match': {'user_id': current_user['_id']}},
        {'$group': {'_id': '$analysis_type', 'count': {'$sum': 1}}}
    ]
    content_types = mongo.db.analyses.aggregate(pipeline)
    result = {item['_id']: item['count'] for item in content_types}
    return jsonify(result), 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)