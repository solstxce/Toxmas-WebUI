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
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip
from nudenet import NudeDetector
from PIL import Image
from pymongo.errors import ConnectionFailure
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import RobertaTokenizer, RobertaModel
from werkzeug.security import generate_password_hash, check_password_hash
from docx import Document
import moviepy.config as cfg
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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
tokenizer = RobertaTokenizer.from_pretrained('./tokenizer')
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

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

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

def categorize_document(abusive_count, total_count):
    percentage = (abusive_count / total_count) * 100
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

class VideoProcessor:
    def __init__(self):
        self.detector = NudeDetector()

    def process_video(self, input_path):
        logger.info(f"Processing video: {input_path}")
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out_path = self._get_output_path(input_path, 'censored_video.mp4')
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            self._process_frames(cap, out, total_frames)
            
            cap.release()
            out.release()
            
            self._add_watermark(out_path, input_path)
            
            logger.info(f"Video processing complete. Output saved to: {out_path}")
            return out_path
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals():
                out.release()

    def _process_frames(self, cap, out, total_frames):
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            logger.debug(f"Processing frame {frame_count}/{total_frames}")
            try:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                temp_frame_path = f'temp_frame_{frame_count}.jpg'
                pil_image.save(temp_frame_path)
                
                censored_img_path = self.detector.censor(temp_frame_path)
                if censored_img_path:
                    censored_frame = cv2.cvtColor(cv2.imread(censored_img_path), cv2.COLOR_BGR2RGB)
                    frame = censored_frame
                else:
                    logger.warning(f"Censorship returned None for frame {frame_count}")
                
                os.remove(temp_frame_path)
                if censored_img_path and os.path.exists(censored_img_path):
                    os.remove(censored_img_path)
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

    def _add_watermark(self, video_path, input_path):
        logger.info("Adding watermark to video")
        video = VideoFileClip(video_path)
        watermark = (TextClip("Verified by NoHate", fontsize=24, color='white', font='Arial')
                     .set_position(('right', 'bottom'))
                     .set_duration(video.duration))
        
        final_video = CompositeVideoClip([video, watermark])
        final_output_path = self._get_output_path(input_path, 'final_censored_video.mp4')
        final_video.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

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

def create_pdf_output(output_path, content, abusive_words, document_state, total_words, pie_chart_path):
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
        is_abusive = any(abusive_word.lower() == word.lower() for abusive_word, _, _ in abusive_words)
        
        if x + can.stringWidth(word + " ") > width - 100:
            y -= 20
            x = 100
            if y < 50:
                can.showPage()
                y = height - 50
        
        can.saveState()
        if is_abusive:
            can.setFillColor(colors.red)
            can.setFont("Helvetica-Bold", 12)
            can.drawString(x, y, word)
            can.setFillColor(colors.yellow)
            can.rect(x, y - 2, can.stringWidth(word), 14, fill=1)
            can.setFillColor(colors.red)
            can.drawString(x, y, word)
            can.line(x, y-2, x + can.stringWidth(word), y-2)
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

    with open(output_path, "wb") as output_stream:
        output.write(output_stream)
    
    os.unlink(pie_chart_path)
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
            content = extract_text(file)
            logger.info("Analyzing extracted text")
            
            words = preprocess_text(content)
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

            output_filename = f"Report for {os.path.splitext(file.filename)[0]}.pdf"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            print(            
                content,
                abusive_words,
                document_state,
                total_words,
                pie_chart_path,
                output_path=output_path,
            )
            create_pdf_output(
                content,
                abusive_words,
                document_state,
                total_words,
                pie_chart_path,
                output_path=output_path,
            )
            
            return send_file(output_path, as_attachment=True, download_name=output_filename, mimetype='application/pdf')
        elif ext.lower() in ['.jpg', '.jpeg', '.png']:
        # elif ext.lower() in ['.jpg', '.jpeg', '.png']:
            logger.info("Processing image file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                logger.info("Analyzing image")
                video_processor = VideoProcessor()
                image_stats = video_processor.analyze_image(temp_file.name)
            os.unlink(temp_file.name)
            return jsonify(image_stats), 200
        elif ext.lower() in ['.mp4', '.avi', '.mov']:
            logger.info("Processing video file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                logger.info("Censoring video")
                video_processor = VideoProcessor()
                censored_video_path = video_processor.process_video(temp_file.name)
            logger.info("Sending censored video")
            return send_file(censored_video_path, as_attachment=True, download_name='censored_video.mp4', mimetype='video/mp4')
        else:
            logger.error(f"Unsupported file format: {ext}")
            return jsonify({'message': 'Unsupported file format'}), 400
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return jsonify({'message': 'An error occurred during analysis'}), 500
if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)