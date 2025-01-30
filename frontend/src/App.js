import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('resume', file);

    try {
      setLoading(true);
      setError('');
      const response = await axios.post('/analyze', formData);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">Career Path Analyzer</h1>
        <div className="upload-section">
          <label className="upload-btn">
            Upload Resume
            <input 
              type="file" 
              onChange={handleFileUpload} 
              accept=".pdf,.doc,.docx" 
              hidden 
            />
          </label>
          <p className="file-types">Supported formats: PDF, DOC, DOCX</p>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p className="loading-text">Analyzing your resume...</p>
        </div>
      )}

      {results && (
        <div className="results-container">
          <div className="recommendations-section">
            <h2 className="section-title">Top Career Matches</h2>
            <div className="recommendations-list">
              {results.recommendations.map((job, index) => (
                <div key={index} className="job-card">
                  <h3 className="job-title">{job.job}</h3>
                  <div className="confidence-container">
                    <div className="confidence-meter">
                      <div 
                        className="confidence-fill" 
                        style={{ width: job.confidence }}
                      ></div>
                    </div>
                    <span className="confidence-value">{job.confidence}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="insights-section">
            <h2 className="section-title">Resume Analysis</h2>
            
            <div className="strengths-container">
              <h3 className="insight-subtitle">
                <span role="img" aria-label="strengths">✅</span> Your Strong Skills
              </h3>
              <div className="skills-grid">
                {results.insights.strengths.map((skill, i) => (
                  <span key={i} className="skill-tag">{skill}</span>
                ))}
              </div>
            </div>

            <div className="improvements-container">
              <h3 className="insight-subtitle">
                <span role="img" aria-label="improvements">📈</span> Recommended Improvements
              </h3>
              <ul className="improvements-list">
                {results.insights.missing_skills.map((skill, i) => (
                  <li key={i} className="improvement-item">{skill}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;