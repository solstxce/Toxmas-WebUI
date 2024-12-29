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
from flask import Flask, jsonify, request, send_file, redirect, url_for, session
from flask_cors import CORS
from flask_pymongo import PyMongo
from flasgger import Swagger, swag_from
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip
from nudenet import NudeDetector
from PIL import Image, ImageDraw
from pymongo.errors import ConnectionFailure
from PyPDF2 import PdfReader
from reportlab.lib.units import inch
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
from flask import Flask, jsonify, request, send_file, redirect, url_for, session
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
from typing import List, Dict, Optional, Union, Any
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from urllib.parse import urlencode
from werkzeug.exceptions import HTTPException
# import env
# Load environment variables
load_dotenv()

# Auth0 configuration
device = 'cpu'

# Define the model class (unchanged)
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Initialize the model, load state_dict, and tokenizer (unchanged)
model = RobertaClass()
model.load_state_dict(torch.load(r'C:\Projects\AntiCyberBullying\Document_Analyzer\model_and_tokenizer\pytorch_roberta_cyberbullying.bin', map_location=torch.device('cpu')))
model.to(device)
tokenizer = RobertaTokenizer.from_pretrained(r'C:\Projects\AntiCyberBullying\Document_Analyzer\model_and_tokenizer\tokenizer')

# Get the set of stopwords (unchanged)
stop_words = set(stopwords.words('english'))

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'profile' not in session:
            return jsonify({'message': 'Authentication required'}), 401
            
        # Get user from MongoDB
        user = mongo.db.users.find_one({'auth0_id': session['profile']['user_id']})
        if not user:
            return jsonify({'message': 'User not found'}), 404
            
        return f(user, *args, **kwargs)
    return decorated

cfg.change_settings({"IMAGEMAGICK_BINARY": "magick.exe"})
import moviepy.config as cfg
from better_profanity import profanity
from nltk.corpus import wordnet
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import moviepy.editor as mp
# Initialize profanity filter
profanity.load_censor_words()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
oauth = OAuth(app)
auth0 = oauth.register(
    'auth0',
    client_id=os.environ.get('AUTH0_CLIENT_ID'),
    client_secret=os.environ.get('AUTH0_CLIENT_SECRET'),
    api_base_url=f'https://{os.environ.get("AUTH0_DOMAIN")}',
    access_token_url=f'https://{os.environ.get("AUTH0_DOMAIN")}/oauth/token',
    authorize_url=f'https://{os.environ.get("AUTH0_DOMAIN")}/authorize',
    client_kwargs={
        'scope': 'openid profile email',
    },
)
CORS(app, supports_credentials=True)
Swagger(app)

# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/document_analyzer'
mongo = PyMongo(app)

# RoBERTa model setup

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


# def predict(text, model, tokenizer, max_len=256):
#     model.eval()
    
#     inputs = tokenizer.encode_plus(
#         text,
#         None,
#         add_special_tokens=True,
#         max_length=max_len,
#         pad_to_max_length=True,
#         return_token_type_ids=True,
#         return_tensors='pt'
#     )
    
#     ids = inputs['input_ids'].to('cpu', dtype=torch.long)
#     mask = inputs['attention_mask'].to('cpu', dtype=torch.long)

#     with torch.no_grad():
#         outputs = model(ids, mask)
    
#     probabilities = F.softmax(outputs, dim=1)
#     _, predicted = torch.max(probabilities, 1)
    
#     predicted_class = predicted.item()
#     predicted_probability = probabilities[0][predicted_class].item()
    
#     return predicted_class, predicted_probability

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
    profane_words = set()
    profane_word_count = 0
    
    for word in words:
        if profanity.contains_profanity(word):
            profane_words.add(word)
            profane_word_count += 1
    
    document_state = categorize_document(len(profane_words), total_words)
    pie_chart_path = create_pie_chart(len(profane_words), total_words)
    
    logger.info(f"Text analysis complete. Document state: {document_state}")
    return document_state, list(profane_words), total_words, profane_word_count, pie_chart_path

def generate_report(document_state, profane_words, total_words, profane_word_count, pie_chart_path, content, input_filename):
    output_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(ANALYSIS_REPORTS_DIR, output_filename)
    
    create_pdf_output(output_path, content, profane_words, total_words, profane_word_count, input_filename)
    
    return output_path

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

ALLOWED_CLASSES = [
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "BELLY_EXPOSED"
]



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

    def process_video(self, input_path, processing_mode='both'):
        """
        Process video with specified mode
        Args:
            input_path: Path to input video
            processing_mode: One of 'audio_only', 'video_only', or 'both' (default)
        """
        logger.info(f"Processing video: {input_path} with mode: {processing_mode}")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initial video setup
                video = mp.VideoFileClip(input_path)
                output_path = input_path
                
                if processing_mode in ['video_only', 'both']:
                    # Video processing
                    frames_dir = os.path.join(temp_dir, 'frames')
                    censored_frames_dir = os.path.join(temp_dir, 'censored_frames')
                    os.makedirs(frames_dir, exist_ok=True)
                    os.makedirs(censored_frames_dir, exist_ok=True)

                    video_info = self._get_video_info(input_path)
                    frame_rate = video_info['frame_rate']
                    duration = video_info['duration']
                    total_frames = int(frame_rate * duration)

                    self._extract_frames(input_path, frames_dir, frame_rate)
                    self._process_frames(frames_dir, censored_frames_dir, total_frames)

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = self._get_output_path(input_path, f'censored_video_{timestamp}.mp4')
                    self._reconstruct_video(censored_frames_dir, output_path, input_path, frame_rate)
                    video = mp.VideoFileClip(output_path)

                if processing_mode in ['audio_only', 'both']:
                    # Audio processing
                    output_path = self._process_audio_profanity(
                        output_path if processing_mode == 'both' else input_path
                    )
                    video = mp.VideoFileClip(output_path)

                # Add watermark regardless of mode
                final_output_path = self._add_watermark(output_path, input_path)
                
                # Clean up
                video.close()

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
        
        def filter_detections(detections):
            """Filter detections based on allowed classes and confidence threshold"""
            return [
                detection for detection in detections
                if (detection.get('class') in ALLOWED_CLASSES and
                    detection.get('score', 0) >= 0.20)  # 60% confidence threshold
            ]

        def custom_censor(image_path, detections):
            """Apply custom censoring to detected areas"""
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            for detection in detections:
                box = detection['box']
                x0, y0, w, h = [int(coord) for coord in box]
                x1 = x0 + w  # Convert width to right coordinate
                y1 = y0 + h  # Convert height to bottom coordinate
                draw.rectangle([x0, y0, x1, y1], fill='black')

            return image

        for i, frame_path in enumerate(frame_files):
            try:
                # Get and filter detections
                detections = self.detector.detect(frame_path)
                filtered_detections = filter_detections(detections)

                # Apply censoring if detections found, otherwise keep original frame
                if filtered_detections:
                    logger.debug(f"Found {len(filtered_detections)} detections in frame {i+1}")
                    censored_img = custom_censor(frame_path, filtered_detections)
                    frame_name = os.path.basename(frame_path)
                    censored_frame_path = os.path.join(censored_frames_dir, frame_name)
                    censored_img.save(censored_frame_path)
                else:
                    # If no detections, copy original frame
                    shutil.copy(frame_path, censored_frames_dir)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{total_frames} frames")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {str(e)}")
                # If processing fails, copy the original frame
                shutil.copy(frame_path, censored_frames_dir)

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

    def _process_audio_profanity(self, video_path: str) -> str:
        """
        Process the video for profanity and mute inappropriate content
        """
        logger.info("Processing audio for profanity")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract audio
                video = mp.VideoFileClip(video_path)
                temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
                
                if video.audio is None:
                    logger.warning("Video has no audio track")
                    video.close()
                    return video_path

                video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

                # Transcribe audio
                result = self.whisper_model.transcribe(
                    temp_audio_path,
                    language="en",
                    word_timestamps=True,
                    verbose=False
                )

                # Detect profanity
                profanity_instances = self._detect_profanity_instances(result)
                
                if not profanity_instances:
                    logger.info("No profanity detected in the video")
                    video.close()
                    return video_path

                # Process audio with pydub
                audio = AudioSegment.from_wav(temp_audio_path)
                processed_audio = self._mute_profanity_segments(audio, profanity_instances)

                # Export processed audio
                temp_processed_audio = os.path.join(temp_dir, "processed_audio.wav")
                processed_audio.export(temp_processed_audio, format="wav")

                # Create output video with processed audio
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self._get_output_path(video_path, f'audio_processed_{timestamp}.mp4')
                processed_audio_clip = mp.AudioFileClip(temp_processed_audio)
                final_video = video.set_audio(processed_audio_clip)

                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(temp_dir, 'temp-final-audio.m4a'),
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )

                # Cleanup
                video.close()
                final_video.close()
                processed_audio_clip.close()

                return output_path

        except Exception as e:
            logger.error(f"Error processing audio for profanity: {str(e)}")
            return video_path

    def _detect_profanity_instances(self, transcription: dict) -> List[Dict]:
        """
        Detect profanity in transcription with timestamps
        """
        profanity_instances = []
        
        for segment in transcription["segments"]:
            if "words" not in segment:
                continue
                
            for word in segment["words"]:
                word_text = word["word"].strip().lower()
                if profanity.contains_profanity(word_text):
                    instance = {
                        "word": word["word"],
                        "start_time": int(word["start"] * 1000),  # Convert to milliseconds
                        "end_time": int(word["end"] * 1000)
                    }
                    profanity_instances.append(instance)
        
        return sorted(profanity_instances, key=lambda x: x["start_time"])

    def _mute_profanity_segments(self, audio: AudioSegment, profanity_instances: List[Dict]) -> AudioSegment:
        """
        Mute sections of audio containing profanity
        """
        for instance in profanity_instances:
            start_time = max(0, instance["start_time"] - 100)  # 100ms buffer
            end_time = min(len(audio), instance["end_time"] + 100)
            
            # Calculate duration and create appropriate silence
            duration = end_time - start_time
            mute_segment = AudioSegment.silent(duration=duration)
            
            # Replace the segment with silence
            audio = audio[:start_time] + mute_segment + audio[end_time:]
        
        return audio

    def _add_watermark(self, video_path, input_path):
        logger.info("Adding watermark to video")
        video = VideoFileClip(video_path)
        watermark = (TextClip("Verified by TOX-MAS", fontsize=24, color='white', font='Arial')
                     .set_position(('right', 'bottom'))
                     .set_duration(video.duration))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_video = CompositeVideoClip([video, watermark])
        final_output_path = self._get_output_path(input_path, f'final_censored_video_{timestamp}.mp4')
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

def check_and_replace_profanity(text):
    contains_profanity = profanity.contains_profanity(text)
    replaced_text = replace_profanity_in_text(text)
    return contains_profanity, replaced_text

# Create a directory for analysis reports
ANALYSIS_REPORTS_DIR = r"C:\Users\Solstxce\Projects\CSP_Project_NoHate\Document-Analyzer V1\analysis_reports" or os.path.join(os.getcwd(), 'analysis_reports')
os.makedirs(ANALYSIS_REPORTS_DIR, exist_ok=True)

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
                _, processed_text = check_and_replace_profanity(text)
                report_path = generate_rephrased_report(processed_text, file.filename)
            else:  # Default to censoring
                logger.info("Analyzing and censoring text")
                document_state, profane_words, total_words, profane_word_count, pie_chart_path = analyze_text(text)
                report_path = generate_report(document_state, profane_words, total_words, profane_word_count, pie_chart_path, text, file.filename)
            
            # Update the analysis_result
            analysis_result = {
                'user_id': current_user['_id'],
                'original_filename': file.filename,
                'result_filename': os.path.basename(report_path),
                'result_path': report_path,
                'analysis_type': 'document',
                'action': action,
                'created_at': datetime.utcnow()
            }
            inserted_id = mongo.db.analyses.insert_one(analysis_result).inserted_id
            
            return jsonify({
                'id': str(inserted_id),
                'result_path': f'/download_analysis/{inserted_id}',
                'message': 'Document analysis completed successfully'
            }), 200

        elif ext.lower() in ['.jpg', '.jpeg', '.png']:
            logger.info("Processing image file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                logger.info("Analyzing image")
                video_processor = VideoProcessor()
                image_stats = video_processor.analyze_image(temp_file.name)
            os.unlink(temp_file.name)
            
            # Update the analysis_result
            analysis_result = {
                'user_id': current_user['_id'],
                'original_filename': file.filename,
                'result_filename': 'censored_image.png',
                'result_path': image_stats['censored_image_path'],
                'analysis_type': 'image',
                'created_at': datetime.utcnow()
            }
            inserted_id = mongo.db.analyses.insert_one(analysis_result).inserted_id
            
            return jsonify({
                'id': str(inserted_id),
                'result_path': f'/download_analysis/{inserted_id}',
                'message': 'Image analysis completed successfully',
                'stats': image_stats
            }), 200

        elif ext.lower() in ['.mp4', '.avi', '.mov']:
            logger.info("Processing video file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                
                # Get processing mode from request
                processing_mode = request.form.get('action', 'both')  # Default to 'both' if not specified
                
                logger.info("Analyzing and censoring video")
                video_processor = VideoProcessor()
                censored_video_path = video_processor.process_video(temp_file.name, processing_mode)

            # Update the analysis_result
            analysis_result = {
                'user_id': current_user['_id'],
                'original_filename': file.filename,
                'result_filename': 'censored_video.mp4',
                'result_path': censored_video_path,
                'analysis_type': 'video',
                'created_at': datetime.utcnow()
            }
            inserted_id = mongo.db.analyses.insert_one(analysis_result).inserted_id
            
            return jsonify({
                'id': str(inserted_id),
                'result_path': f'/download_analysis/{inserted_id}',
                'message': 'Video analysis completed successfully'
            }), 200

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
            'created_at': analysis['created_at'].isoformat(),
            'result_path': f'/download_analysis/{analysis["_id"]}'
        })
    return jsonify(result), 200

# Add a new route to download analysis results
@app.route('/download_analysis/<analysis_id>', methods=['GET','POST'])
@token_required
def download_analysis(current_user, analysis_id):
    analysis = mongo.db.analyses.find_one({'_id': ObjectId(analysis_id), 'user_id': current_user['_id']})
    if not analysis:
        return jsonify({'message': 'Analysis not found'}), 404

    file_path = analysis['result_path']
    return send_file(file_path, as_attachment=True, download_name=analysis['result_filename'])

@app.route('/view_analysis/<analysis_id>', methods=['GET'])
@token_required
def view_analysis(current_user, analysis_id):
    analysis = mongo.db.analyses.find_one({'_id': ObjectId(analysis_id), 'user_id': current_user['_id']})
    if not analysis:
        return jsonify({'message': 'Analysis not found'}), 404

    file_path = analysis['result_path']
    return send_file(file_path, as_attachment=False)
# Add a new function to generate a report for rephrased text
def generate_rephrased_report(rephrased_text, input_filename):
    output_filename = f"rephrased_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(ANALYSIS_REPORTS_DIR, output_filename)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Register a custom font for the watermark
    pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'helvetica-bold.ttf'))
    
    def add_watermark(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 40)
        # canvas.setFont(, 40)
        canvas.setFillColorRGB(0.9, 0.9, 0.9)  # Light gray color
        canvas.translate(inch, inch)
        canvas.rotate(45)
        canvas.drawCentredString(400, 0, "Analyzed by TOX-MAS")
        canvas.restoreState()
    
    content = []
    
    # Add title
    title_style = ParagraphStyle('Title', parent=styles['Title'], alignment=1, spaceAfter=0.3*inch)
    title = Paragraph(f"Rephrased Document Report for {input_filename}", title_style)
    content.append(title)
    content.append(Spacer(1, 0.25*inch))
    
    # Add rephrased text
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], spaceBefore=0.1*inch, spaceAfter=0.1*inch)
    rephrased_paragraphs = rephrased_text.split('\n')
    for paragraph in rephrased_paragraphs:
        content.append(Paragraph(paragraph, normal_style))
        content.append(Spacer(1, 0.1*inch))
    
    # Build the PDF
    doc.build(content, onFirstPage=add_watermark, onLaterPages=add_watermark)
    
    return output_path

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

def create_pdf_output(output_path, content, profane_words, total_words, profane_word_count, input_filename):
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    width, height = letter

    # Register a custom font for the watermark
    pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'helvetica-bold.ttf'))

    # Set up styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], alignment=1, spaceAfter=0.3*inch)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], spaceBefore=0.2*inch, spaceAfter=0.1*inch)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], spaceBefore=0.1*inch, spaceAfter=0.1*inch)

    def add_watermark(canvas):
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 40)
        canvas.setFillColorRGB(0.9, 0.9, 0.9)  # Light gray color
        canvas.translate(inch, inch)
        canvas.rotate(45)
        canvas.drawCentredString(400, 0, "Analyzed by TOX-MAS")
        canvas.restoreState()

    # Add watermark
    add_watermark(can)

    # Add title
    title = Paragraph(f"Document Analysis Report for {input_filename}", title_style)
    title.wrapOn(can, width - 2*inch, height)
    title.drawOn(can, inch, height - inch)

    # Add summary
    summary = [
        Paragraph("Summary:", heading_style),
        Paragraph(f"Total words: {total_words}", normal_style),
        Paragraph(f"Profane words found: {profane_word_count}", normal_style),
        Paragraph(f"Percentage of profane words: {(profane_word_count / total_words) * 100:.2f}%", normal_style),
    ]

    y = height - 2*inch
    for item in summary:
        item.wrapOn(can, width - 2*inch, height)
        item.drawOn(can, inch, y)
        y -= item.height + 0.1*inch

    # Add content heading
    content_heading = Paragraph("Document Content", heading_style)
    content_heading.wrapOn(can, width - 2*inch, height)
    content_heading.drawOn(can, inch, height - 3.5*inch)

    # Add content with censored words
    words = content.split()
    x, y = inch, height - 4*inch

    for word in words:
        is_profane = word in profane_words

        if x + can.stringWidth(word + " ") > width - inch:
            y -= 20
            x = inch
            if y < inch:
                can.showPage()
                add_watermark(can)  # Add watermark to new page
                y = height - inch

        if is_profane:
            can.setFillColor(colors.black)
            can.rect(x, y - 2, can.stringWidth(word), 14, fill=1)
        else:
            can.setFont("Helvetica", 12)
            can.setFillColor(colors.black)
            can.drawString(x, y, word)

        x += can.stringWidth(word + " ")

    # Add footer
    can.setFont("Helvetica", 8)
    can.drawString(inch, 0.5*inch, f"Document Analysis Report - Page 1")
    can.drawRightString(width - inch, 0.5*inch, "Generated by Document Analyzer")

    can.save()

    packet.seek(0)
    new_pdf = PdfReader(packet)
    output = PdfWriter()

    for page in new_pdf.pages:
        output.add_page(page)

    with open(output_path, "wb") as output_stream:
        output.write(output_stream)

@app.route('/callback')
def callback_handling():
    auth0.authorize_access_token()
    resp = auth0.get('userinfo')
    userinfo = resp.json()

    session['jwt_payload'] = userinfo
    session['profile'] = {
        'user_id': userinfo['sub'],
        'name': userinfo.get('name', ''),
        'picture': userinfo.get('picture', ''),
        'email': userinfo.get('email', '')
    }

    # Store or update user in MongoDB
    user_data = {
        'auth0_id': userinfo['sub'],
        'email': userinfo.get('email', ''),
        'name': userinfo.get('name', ''),
        'picture': userinfo.get('picture', ''),
        'last_login': datetime.now()
    }
    
    mongo.db.users.update_one(
        {'auth0_id': userinfo['sub']},
        {'$set': user_data},
        upsert=True
    )

    return redirect('/dashboard')

# @app.route('/login')
# def login():
#     return auth0.authorize_redirect(
#         redirect_uri=os.environ.get('AUTH0_CALLBACK_URL'),
#         audience=os.environ.get('API_AUDIENCE')
#     )

@app.route('/logout')
def logout():
    session.clear()
    params = {
        'returnTo': url_for('home', _external=True),
        'client_id': os.environ.get('AUTH0_CLIENT_ID')
    }
    return redirect(auth0.api_base_url + '/v2/logout?' + urlencode(params))

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True,use_reloader=False,host='0.0.0.0',port=5000)