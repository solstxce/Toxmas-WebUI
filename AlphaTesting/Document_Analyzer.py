import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from docx import Document
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
import os
import tempfile
from collections import Counter

# Download the stopwords data
nltk.download('stopwords', quiet=True)

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

# List of abusive words (unchanged)
abusive_word_list = [
    "fuck", "shit", "asshole", "bitch", "cunt", "damn", "hell", "bastard", "motherfucker",
    "dick", "pussy", "slut", "whore", "idiot", "stupid.", "dumb", "retard", "loser", "jerk",
    "moron", "imbecile", "cretin", "twat", "wanker", "prick", "dickhead", "arsehole", "fag",
    "faggot", "homo", "queer", "dyke", "lesbo", "tranny", "nigger", "nigga", "spic", "kike","boobs"
    "trash","booty","baddie","chink", "gook", "wetback", "beaner", "gringo", "cracker",
    "redneck", "hillbilly", "white trash", "gay", "lgbt", "lesbians", "sexy", "fucking", "hell",
    "porn", "perv", "fart", "stupid", "idiot", "ass"
]

# Prediction function (unchanged)
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
    
    ids = inputs['input_ids'].to(device, dtype=torch.long)
    mask = inputs['attention_mask'].to(device, dtype=torch.long)
    token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
    
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    predicted_probability = probabilities[0][predicted_class].item()
    
    return predicted_class, predicted_probability

# Other helper functions (unchanged)
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

# Modified analyze_document function
def analyze_document(input_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    input_filename = os.path.basename(input_path)
    output_filename = f"Report for {os.path.splitext(input_filename)[0]}.pdf"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    if file_extension == '.txt':
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
    elif file_extension == '.docx':
        doc = Document(input_path)
        content = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension == '.pdf':
        reader = PdfReader(input_path)
        content = ''
        for page in reader.pages:
            content += page.extract_text()
    else:
        raise ValueError("Unsupported file format")

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

    create_pdf_output(output_path, content, abusive_words, document_state, total_words, pie_chart_path)

# Modified create_pdf_output function
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
# Example usage
input_document_path = r"C:\Projects\AntiCyberBullying\Document_Analyzer\Transcript.txt"  # Change this to your input file
analyze_document(input_document_path)