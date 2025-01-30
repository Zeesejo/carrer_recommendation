from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import docx2txt
import spacy
import numpy as np
import joblib
from transformers import pipeline
import re
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get absolute paths to model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load NLP components
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load ML models and components
try:
    tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    models = {
        'rf': joblib.load(os.path.join(MODELS_DIR, 'randomforest_model.pkl')),
        'svm': joblib.load(os.path.join(MODELS_DIR, 'svc_model.pkl')),
        'logreg': joblib.load(os.path.join(MODELS_DIR, 'logisticregression_model.pkl')),
        'knn': joblib.load(os.path.join(MODELS_DIR, 'kneighbors_model.pkl')),
    }
    
    # Load DistilBERT if available (using POSIX path)
    bert_path = os.path.join(MODELS_DIR, 'distilbert_model').replace('\\', '/')
    if os.path.isdir(bert_path):
        models['distilbert'] = pipeline(
            "text-classification",
            model=bert_path,
            tokenizer=bert_path
        )
        logger.info("DistilBERT model loaded successfully")
    else:
        logger.warning("DistilBERT model not found, proceeding without it")

except FileNotFoundError as e:
    logger.error(f"Model loading error: {str(e)}")
    logger.error("Please ensure you've:")
    logger.error("1. Run the training script (train_models.py)")
    logger.error("2. Have all model files in the backend/models directory")
    exit(1)

# Configure job skills (update with your categories)
JOB_SKILLS = {
    'Python Developer': ['Python', 'Django', 'Flask', 'REST APIs', 'SQL', 'Git'],
    'Java Developer': ['Java', 'Spring', 'Hibernate', 'J2EE', 'Microservices'],
    # Add remaining categories...
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf', 'doc', 'docx'}

def extract_text(file):
    """Extract text from PDF or Word documents"""
    try:
        if file.filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages])
        elif file.filename.endswith(('.doc', '.docx')):
            return docx2txt.process(file)
        raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        raise

def clean_resume(text):
    """Clean and preprocess resume text"""
    text = re.sub(r'http\S+\s*', ' ', text)  # Remove URLs
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

@app.route('/')
def home():
    """API status and documentation"""
    return jsonify({
        "status": "API Operational",
        "version": "1.1.0",
        "endpoints": {
            "POST /analyze": {
                "description": "Analyze resume and get career recommendations",
                "parameters": {
                    "resume": "PDF/DOC/DOCX file upload"
                }
            }
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """Main analysis endpoint"""
    try:
        # Validate file upload
        if 'resume' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only PDF/DOC/DOCX allowed"}), 400

        # Process file
        raw_text = extract_text(file)
        cleaned_text = clean_resume(raw_text)
        
        # Feature transformation
        tfidf_features = tfidf.transform([cleaned_text])
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            try:
                if model_name == 'distilbert':
                    result = model(cleaned_text)[0]
                    predictions[model_name] = {
                        'label': result['label'],
                        'confidence': result['score']
                    }
                else:
                    proba = model.predict_proba(tfidf_features)[0]
                    top_idx = np.argmax(proba)
                    predictions[model_name] = {
                        'label': le.inverse_transform([top_idx])[0],
                        'confidence': float(proba[top_idx])
                    }
            except Exception as model_error:
                logger.error(f"Error in {model_name} prediction: {str(model_error)}")
                continue

        # Hybrid ensemble with weighted average
        hybrid_scores = {}
        weights = {
            'rf': 0.3,
            'svm': 0.3,
            'logreg': 0.2,
            'knn': 0.1,
            'distilbert': 0.1 if 'distilbert' in models else 0
        }
        
        for model_name, pred in predictions.items():
            label = pred['label']
            hybrid_scores[label] = hybrid_scores.get(label, 0) + weights[model_name] * pred['confidence']
        
        # Get top 3 recommendations
        recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate insights
        doc = nlp(cleaned_text)
        print(doc.ents)
        print("-----------------")  
        print([ ent.label_ for ent in doc.ents])
        user_skills = {ent.text for ent in doc.ents if ent.label_ == "skills"}
        top_category = recommendations[0][0] if recommendations else None
        
        missing_skills = []
        if top_category:
            missing_skills = list(set(JOB_SKILLS.get(top_category, [])) - user_skills)
        
        return jsonify({
            'recommendations': [
                {'job': job, 'confidence': f"{conf*100:.1f}%"} 
                for job, conf in recommendations
            ],
            'insights': {
                'strengths': sorted(user_skills),
                'missing_skills': missing_skills,
                'summary': f"Top recommendation: {top_category} ({recommendations[0][1]*100:.1f}% confidence)" if top_category else "No clear recommendation"
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Analysis failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)