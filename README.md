# Document Analyzer

A secure document analysis platform powered by AI that provides advanced document processing, morphing detection, and content analysis capabilities.

## Features

- **Document Analysis**: Process and analyze various document formats
- **Morph Detection**: Advanced AI-powered image morphing detection
- **Content Security**: Identify and protect sensitive information
- **Multi-format Support**: Handle various file types including images, PDFs, and documents
- **Real-time Processing**: Instant analysis and results
- **Secure Authentication**: JWT-based user authentication
- **Dark Mode Support**: Comfortable viewing in any lighting condition
- **Responsive Design**: Works seamlessly across devices

## Tech Stack

### Frontend
- HTML5/CSS3
- TailwindCSS
- Alpine.js
- Plotly.js

### Backend
- Python 3.x
- Flask
- MongoDB
- PyTorch
- OpenCV
- Transformers (Hugging Face)
- JWT Authentication

## Prerequisites

- Python 3.8+
- MongoDB
- Node.js (for npm packages)
- Git LFS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-analyzer.git
cd document-analyzer
```

2. Install Git LFS and pull the model files:
```bash
git lfs install
git lfs pull
```

3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. Create necessary directories:
```bash
mkdir analysis_reports
mkdir model_and_tokenizer
```

## Configuration

Create a `.env` file in the root directory with the following variables:
```env
FLASK_APP=final_backend.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
MONGO_URI=mongodb://localhost:27017/document_analyzer
JWT_SECRET_KEY=your_jwt_secret
```

## Running the Application

1. Start MongoDB:
```bash
mongod
```

2. Run the Flask application:
```bash
python final_backend.py
```

3. Access the application at `http://localhost:5000`

## Project Structure

```
document-analyzer/
├── final_backend.py        # Main Flask application
├── templates/             # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── index.html
│   ├── login.html
│   └── register.html
├── model_and_tokenizer/   # AI model files
├── analysis_reports/      # Generated reports
├── static/               # Static assets
└── requirements.txt      # Python dependencies
```

## API Endpoints

- `POST /login` - User authentication
- `POST /register` - User registration
- `POST /analyze` - Document analysis
- `POST /process_morphing` - Morph detection
- `GET /download_analysis/<id>` - Download analysis results

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TailwindCSS](https://tailwindcss.com/)
- [Alpine.js](https://alpinejs.dev/)
- [Flask](https://flask.palletsprojects.com/)
- [MongoDB](https://www.mongodb.com/)
- [PyTorch](https://pytorch.org/)
