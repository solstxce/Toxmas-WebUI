import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F
import re
import nltk
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
from fuzzywuzzy import process
import profanity_list
# Download required NLTK data
nltk.download('punkt', quiet=True)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model class
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

# Initialize the model, load state_dict, and tokenizer
model = RobertaClass()
model.load_state_dict(torch.load('./model_and_tokenizer/pytorch_roberta_cyberbullying.bin', map_location=torch.device(device)))
model.to(device)
tokenizer = RobertaTokenizer.from_pretrained('./model_and_tokenizer/tokenizer')

# Define abusive word list and phrases
abusive_word_list = profanity_list.wordlist

abusive_phrases = {
    "go to hell", "f off", "shut up", "drop dead", "son of a bitch", "piece of shit",
    # Add more phrases as needed
}

# Prediction function
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

# Improved text preprocessing
def preprocess_text(text):
    # Remove punctuation, but keep apostrophes for contractions
    text = re.sub(r'[^\w\s\']', '', text)
    return text.lower().split()

# Fuzzy matching function
def fuzzy_profanity_check(word, abusive_words, threshold=80):
    match = process.extractOne(word, abusive_words)
    return match[1] >= threshold, match[0]

# Phrase detection function
def detect_profanity(text, abusive_phrases):
    words = preprocess_text(text)
    profanity_found = []
    for i in range(len(words)):
        for j in range(1, min(5, len(words) - i + 1)):  # Check phrases up to 4 words long
            phrase = ' '.join(words[i:i+j])
            if phrase in abusive_phrases:
                profanity_found.append((phrase, i, i+j))
    return profanity_found

# Analyze text using the ML model
def analyze_text(text, model, tokenizer, threshold=0.5):
    words = preprocess_text(text)
    abusive_words = []
    for i in range(len(words)):
        for j in range(1, min(5, len(words) - i + 1)):
            phrase = ' '.join(words[i:i+j])
            _, probability = predict(phrase, model, tokenizer)
            if probability > threshold:
                abusive_words.append((phrase, probability))
    return abusive_words

# Comprehensive profanity check
def comprehensive_profanity_check(text, abusive_word_list, abusive_phrases, model, tokenizer):
    word_list_results = detect_profanity(text, abusive_phrases)
    fuzzy_results = []
    for word in preprocess_text(text):
        is_profane, matched_word = fuzzy_profanity_check(word, abusive_word_list)
        if is_profane:
            fuzzy_results.append((word, matched_word))
    ml_results = analyze_text(text, model, tokenizer)
    
    # Combine and deduplicate results
    all_results = set([(w, w) for w, _, _ in word_list_results] + 
                      fuzzy_results + 
                      [(w, w) for w, _ in ml_results])
    return list(all_results)

# Helper functions
def categorize_document(abusive_count, total_count):
    percentage = (abusive_count / total_count) * 100
    if percentage < 2:
        return "Good"
    elif 2 <= percentage <= 5:
        return "Average"
    else:
        return "Bad"

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

# Analyze document function
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
    
    abusive_words = comprehensive_profanity_check(content, abusive_word_list, abusive_phrases, model, tokenizer)
    abusive_word_counts = Counter(word.lower() for word, _ in abusive_words)
    
    document_state = categorize_document(len(abusive_words), total_words)
    pie_chart_path = create_pie_chart(len(abusive_words), total_words)

    create_pdf_output(output_path, content, abusive_words, document_state, total_words, pie_chart_path)

# Create PDF output function
def create_pdf_output(output_path, content, abusive_words, document_state, total_words, pie_chart_path):
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    width, height = letter

    can.drawString(100, height - 50, "Document Analysis")
    can.drawImage(pie_chart_path, 100, height - 300, width=300, height=200)
    can.drawString(100, height - 350, f"Document State: {document_state}")
    can.drawString(100, height - 370, f"Total words: {total_words}")
    can.drawString(100, height - 390, f"Number of potentially abusive words/phrases found: {len(abusive_words)}")
    can.drawString(100, height - 410, f"Percentage of potentially abusive content: {(len(abusive_words) / total_words) * 100:.2f}%")
    
    y = height - 450
    can.drawString(100, y, "Potentially abusive words/phrases and their matched forms:")
    y -= 20
    for word, matched_form in abusive_words:
        can.drawString(120, y, f"{word} (matched: {matched_form})")
        y -= 20
        if y < 50:
            can.showPage()
            y = height - 50

    can.showPage()

    words = content.split()
    x, y = 100, height - 50
    for word in words:
        is_abusive = any(abusive_word.lower() == word.lower() for abusive_word, _ in abusive_words)
        
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

# Main execution
if __name__ == "__main__":
    input_document_path = "./sample.txt"  # Change this to your input file
    analyze_document(input_document_path)
    print(f"Analysis complete. Report saved as 'Report for {os.path.splitext(os.path.basename(input_document_path))[0]}.pdf'")