import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import joblib

# Path configuration
DATA_PATH = "E:/Sem 7/mjp/career-recommender/backend/job_database/UpdatedResumeDataSet.csv"
MODEL_DIR = "E:/Sem 7/mjp/career-recommender/backend/models"
REPORT_DIR = "E:/Sem 7/mjp/career-recommender/backend/reports"

def clean_resume(text):
    """Clean resume text with enhanced pattern matching"""
    patterns = [
        (r'http\S+\s*', ' '),          # URLs
        (r'RT|cc', ' '),               # Social media tags
        (r'#\S+', ''),                 # Hashtags
        (r'@\S+', ' '),                # Mentions
        (r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' '),  # Punctuation
        (r'[^\x00-\x7f]', ' '),        # Non-ASCII
        (r'\s+', ' '),                 # Extra whitespace
        (r'\b\d+\b', ' ')              # Standalone numbers
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text.strip()

def main():
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    df['cleaned_resume'] = df['Resume'].apply(clean_resume)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(df['cleaned_resume'])
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class distribution analysis
    print("\nClass Distribution Analysis:")
    print(f"Original: {np.bincount(y)}")
    print(f"Training: {np.bincount(y_train)}")
    print(f"Testing:  {np.bincount(y_test)}")

    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
        'SVC': SVC(kernel='linear', class_weight='balanced', probability=True),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced')
    }

    # Training and evaluation
    results = []
    for name, model in models.items():
        print(f"\n{'='*40}\nEvaluating {name}\n{'='*40}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"CV Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Full training
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        
        # Store results
        results.append({
            'model': name,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        })

        # Save model
        joblib.dump(model, os.path.join(MODEL_DIR, f'{name.lower()}_model.pkl'))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(xticks_rotation=90)
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(os.path.join(REPORT_DIR, f'{name.lower()}_cm.png'), bbox_inches='tight')
        plt.close()

    # Save final report
    report_df = pd.DataFrame(results)
    print("\nFinal Report:")
    print(report_df)
    report_df.to_csv(os.path.join(REPORT_DIR, 'model_performance.csv'), index=False)

    # Feature importance (for RandomForest)
    if 'RandomForest' in models:
        importances = models['RandomForest'].feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        print("\nTop 20 Important Features:")
        print([feature_names[i] for i in top_idx])

    # Sample prediction test
    test_resume = """
    Experienced software developer with 5+ years in full-stack development.
    Skills: Python, JavaScript, React, Node.js, SQL, Docker, AWS.
    Education: BS in Computer Science from State University.
    """
    test_clean = clean_resume(test_resume)
    test_vec = vectorizer.transform([test_clean])
    
    print("\nSample Prediction Test:")
    for name, model in models.items():
        pred = le.inverse_transform(model.predict(test_vec))[0]
        proba = np.max(model.predict_proba(test_vec))
        print(f"{name}: {pred} ({proba:.2%})")

if __name__ == "__main__":
    main()