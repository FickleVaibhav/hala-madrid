"""
Streamlit Web Application for Respiratory Disease Detection
=========================================================

This application provides a user-friendly interface for recording lung sounds,
analyzing audio signals, and predicting respiratory diseases using machine learning.
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import tempfile
import os
from datetime import datetime
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.preprocessing.audio_preprocessing import AudioPreprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.models.ensemble_model import EnsembleModel
from src.utils.visualization import AudioVisualizer
from src.utils.config import Config

# Page configuration
st.set_page_config(
    page_title="Respiratory Disease Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #424242;
    border-bottom: 2px solid #e3f2fd;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}
.prediction-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid;
}
.normal { border-left-color: #4caf50; background-color: #e8f5e8; }
.abnormal { border-left-color: #f44336; background-color: #ffebee; }
.warning { border-left-color: #ff9800; background-color: #fff3e0; }
.info-box {
    padding: 1rem;
    background-color: #f5f5f5;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.metric-container {
    display: flex;
    justify-content: space-around;
    margin: 1rem 0;
}
.metric-item {
    text-align: center;
    padding: 1rem;
    background-color: white;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

class RespiratoryDiseaseApp:
    """Main application class for respiratory disease detection."""
    
    def __init__(self):
        self.config = Config()
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.visualizer = AudioVisualizer()
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained ensemble model."""
        try:
            self.model = EnsembleModel()
            model_path = "data/models/trained/ensemble_model.h5"
            if os.path.exists(model_path):
                self.model.load_model(model_path)
                st.success("✅ Model loaded successfully!")
            else:
                st.warning("⚠️ Pre-trained model not found. Please train the model first.")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🫁 Respiratory Disease Detection</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p><strong>🔬 Research Tool for Lung Sound Analysis</strong></p>
        <p>Upload lung sound recordings or record directly using your smartphone's digital stethoscope 
        to detect respiratory conditions including wheeze, crackle, pneumonia, COPD, and asthma.</p>
        <p><em>⚠️ This tool is for research and educational purposes only. 
        Not intended for clinical diagnosis.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        st.sidebar.title("🎛️ Controls")
        
        # Audio input selection
        input_method = st.sidebar.selectbox(
            "Select Input Method",
            ["Upload Audio File", "Record Live Audio", "Use Demo Sample"]
        )
        
        # Model selection
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["Ensemble Model (Recommended)", "CNN Model", "ResNet Model"]
        )
        
        # Processing options
        st.sidebar.subheader("⚙️ Processing Options")
        
        apply_noise_reduction = st.sidebar.checkbox("Apply Noise Reduction", value=True)
        normalize_audio = st.sidebar.checkbox("Normalize Audio", value=True)
        
        sample_rate = st.sidebar.selectbox(
            "Sample Rate (Hz)",
            [16000, 22050, 44100, 48000],
            index=0
        )
        
        # Visualization options
        st.sidebar.subheader("📊 Visualization")
        
        show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
        show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)
        show_mfcc = st.sidebar.checkbox("Show MFCC Features", value=False)
        
        return {
            'input_method': input_method,
            'model_type': model_type,
            'apply_noise_reduction': apply_noise_reduction,
            'normalize_audio': normalize_audio,
            'sample_rate': sample_rate,
            'show_waveform': show_waveform,
            'show_spectrogram': show_spectrogram,
            'show_mfcc': show_mfcc
        }
    
    def handle_audio_input(self, options):
        """Handle different audio input methods."""
        audio_data = None
        sample_rate = options['sample_rate']
        
        if options['input_method'] == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload lung sound recording",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Supported formats: WAV, MP3, FLAC, M4A"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_data, sample_rate = librosa.load(tmp_file.name, sr=sample_rate)
                    os.unlink(tmp_file.name)
                
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.info(f"📊 Duration: {len(audio_data)/sample_rate:.2f}s, Sample Rate: {sample_rate}Hz")
        
        elif options['input_method'] == "Record Live Audio":
            st.markdown('<h3 class="section-header">🎙️ Live Audio Recording</h3>', 
                       unsafe_allow_html=True)
            
            # Audio recording interface
            audio_bytes = st.audio_input("Record lung sounds using your smartphone stethoscope")
            
            if audio_bytes is not None:
                # Convert audio bytes to numpy array
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes.getbuffer())
                    audio_data, sample_rate = librosa.load(tmp_file.name, sr=sample_rate)
                    os.unlink(tmp_file.name)
                
                st.success("✅ Audio recorded successfully!")
                st.audio(audio_bytes, format='audio/wav')
        
        elif options['input_method'] == "Use Demo Sample":
            st.markdown('<h3 class="section-header">🎬 Demo Samples</h3>', 
                       unsafe_allow_html=True)
            
            demo_samples = {
                "Normal Breathing": "data/demo/normal_breathing.wav",
                "Wheeze (Asthma)": "data/demo/wheeze_asthma.wav", 
                "Crackle (Pneumonia)": "data/demo/crackle_pneumonia.wav",
                "COPD": "data/demo/copd_sample.wav"
            }
            
            selected_demo = st.selectbox("Select demo sample", list(demo_samples.keys()))
            
            if st.button("Load Demo Sample"):
                demo_path = demo_samples[selected_demo]
                if os.path.exists(demo_path):
                    audio_data, sample_rate = librosa.load(demo_path, sr=sample_rate)
                    st.success(f"✅ Loaded demo: {selected_demo}")
                else:
                    st.warning("⚠️ Demo file not found. Please check the data directory.")
        
        return audio_data, sample_rate
    
    def process_audio(self, audio_data, sample_rate, options):
        """Process the audio data with selected options."""
        if audio_data is None:
            return None
        
        st.markdown('<h3 class="section-header">🔄 Audio Processing</h3>', 
                   unsafe_allow_html=True)
        
        processed_audio = audio_data.copy()
        
        with st.spinner("Processing audio..."):
            # Apply noise reduction
            if options['apply_noise_reduction']:
                processed_audio = self.preprocessor.reduce_noise(processed_audio, sample_rate)
                st.success("✅ Noise reduction applied")
            
            # Normalize audio
            if options['normalize_audio']:
                processed_audio = self.preprocessor.normalize_audio(processed_audio)
                st.success("✅ Audio normalized")
            
            # Segment into respiratory cycles
            cycles = self.preprocessor.segment_respiratory_cycles(processed_audio, sample_rate)
            st.info(f"📊 Detected {len(cycles)} respiratory cycles")
        
        return processed_audio, cycles
    
    def visualize_audio(self, audio_data, sample_rate, options):
        """Create audio visualizations."""
        if audio_data is None:
            return
        
        st.markdown('<h3 class="section-header">📊 Audio Analysis</h3>', 
                   unsafe_allow_html=True)
        
        # Create columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if options['show_waveform']:
                st.subheader("🌊 Waveform")
                fig_wave = self.visualizer.plot_waveform(audio_data, sample_rate)
                st.pyplot(fig_wave)
            
            if options['show_mfcc']:
                st.subheader("🎵 MFCC Features")
                mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
                fig_mfcc = self.visualizer.plot_mfcc(mfcc, sample_rate)
                st.pyplot(fig_mfcc)
        
        with col2:
            if options['show_spectrogram']:
                st.subheader("🎨 Mel Spectrogram")
                fig_spec = self.visualizer.plot_spectrogram(audio_data, sample_rate)
                st.pyplot(fig_spec)
    
    def extract_features(self, audio_data, sample_rate):
        """Extract features for model prediction."""
        if audio_data is None:
            return None
        
        features = self.feature_extractor.extract_comprehensive_features(
            audio_data, sample_rate
        )
        return features
    
    def predict_disease(self, features, options):
        """Predict respiratory disease using the selected model."""
        if features is None or self.model is None:
            return None
        
        st.markdown('<h3 class="section-header">🔮 Disease Prediction</h3>', 
                   unsafe_allow_html=True)
        
        with st.spinner("Analyzing lung sounds..."):
            # Make prediction
            prediction = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
        
        return prediction, probabilities
    
    def display_results(self, prediction, probabilities):
        """Display prediction results with visualization."""
        if prediction is None:
            return
        
        class_names = [
            "Normal", "Wheeze", "Crackle", "Pneumonia", 
            "COPD", "Asthma", "Bronchiectasis", "Other"
        ]
        
        predicted_class = class_names[prediction[0]]
        confidence = np.max(probabilities) * 100
        
        # Display main prediction
        if predicted_class == "Normal":
            box_class = "normal"
            icon = "✅"
        elif confidence > 80:
            box_class = "abnormal" 
            icon = "🚨"
        else:
            box_class = "warning"
            icon = "⚠️"
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
        <h3>{icon} Prediction: {predicted_class}</h3>
        <p><strong>Confidence: {confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display probability distribution
        st.subheader("📊 Probability Distribution")
        
        prob_df = pd.DataFrame({
            'Disease': class_names,
            'Probability': probabilities[0] * 100
        }).sort_values('Probability', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=prob_df['Probability'],
            y=prob_df['Disease'],
            orientation='h',
            text=prob_df['Probability'].round(1).astype(str) + '%',
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Disease Classification Probabilities",
            xaxis_title="Probability (%)",
            yaxis_title="Respiratory Condition",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Medical recommendations
        self.display_recommendations(predicted_class, confidence)
    
    def display_recommendations(self, predicted_class, confidence):
        """Display medical recommendations based on prediction."""
        st.subheader("💡 Recommendations")
        
        recommendations = {
            "Normal": {
                "message": "Lung sounds appear normal. Continue regular health monitoring.",
                "actions": ["Maintain healthy lifestyle", "Regular exercise", "Avoid smoking"]
            },
            "Wheeze": {
                "message": "Wheeze detected. May indicate airway obstruction.",
                "actions": ["Consult pulmonologist", "Check for asthma triggers", "Monitor breathing"]
            },
            "Crackle": {
                "message": "Crackles detected. May indicate fluid in lungs.",
                "actions": ["Seek immediate medical evaluation", "Check for pneumonia", "Monitor symptoms"]
            },
            "Pneumonia": {
                "message": "Signs consistent with pneumonia detected.",
                "actions": ["Urgent medical consultation required", "Consider chest X-ray", "Monitor fever/symptoms"]
            },
            "COPD": {
                "message": "Signs consistent with COPD detected.",
                "actions": ["Pulmonologist consultation", "Spirometry testing", "Smoking cessation if applicable"]
            },
            "Asthma": {
                "message": "Signs consistent with asthma detected.",
                "actions": ["Consult allergist/pulmonologist", "Identify triggers", "Consider rescue inhaler"]
            },
            "Bronchiectasis": {
                "message": "Signs consistent with bronchiectasis detected.",
                "actions": ["Specialized pulmonary evaluation", "CT scan may be needed", "Airway clearance techniques"]
            },
            "Other": {
                "message": "Abnormal sounds detected but classification uncertain.",
                "actions": ["General medical evaluation recommended", "Additional testing may be needed"]
            }
        }
        
        rec = recommendations.get(predicted_class, recommendations["Other"])
        
        if confidence < 70:
            st.warning("⚠️ Low confidence prediction. Results should be interpreted carefully.")
        
        st.info(rec["message"])
        
        st.write("**Suggested Actions:**")
        for action in rec["actions"]:
            st.write(f"• {action}")
        
        st.error("""
        **⚠️ IMPORTANT MEDICAL DISCLAIMER:**
        This tool is for research and educational purposes only. Results should not be used for 
        medical diagnosis or treatment decisions. Always consult qualified healthcare professionals 
        for proper medical evaluation and treatment.
        """)
    
    def display_session_history(self):
        """Display analysis history for the session."""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if st.session_state.analysis_history:
            st.markdown('<h3 class="section-header">📜 Session History</h3>', 
                       unsafe_allow_html=True)
            
            history_df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Download history as CSV
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name=f"respiratory_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def run(self):
        """Main application runner."""
        self.render_header()
        
        # Get sidebar options
        options = self.render_sidebar()
        
        # Handle audio input
        audio_data, sample_rate = self.handle_audio_input(options)
        
        if audio_data is not None:
            # Process audio
            processed_audio, cycles = self.process_audio(audio_data, sample_rate, options)
            
            # Visualize audio
            self.visualize_audio(processed_audio, sample_rate, options)
            
            # Extract features
            features = self.extract_features(processed_audio, sample_rate)
            
            # Make prediction
            if st.button("🔬 Analyze Lung Sounds", type="primary"):
                prediction, probabilities = self.predict_disease(features, options)
                
                if prediction is not None:
                    # Display results
                    self.display_results(prediction, probabilities)
                    
                    # Add to session history
                    if 'analysis_history' not in st.session_state:
                        st.session_state.analysis_history = []
                    
                    class_names = [
                        "Normal", "Wheeze", "Crackle", "Pneumonia", 
                        "COPD", "Asthma", "Bronchiectasis", "Other"
                    ]
                    
                    st.session_state.analysis_history.append({
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Input Method': options['input_method'],
                        'Prediction': class_names[prediction[0]],
                        'Confidence': f"{np.max(probabilities) * 100:.1f}%",
                        'Duration': f"{len(processed_audio)/sample_rate:.2f}s"
                    })
        
        # Display session history
        self.display_session_history()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔬 Respiratory Disease Detection System | Research Tool v1.0</p>
        <p>Built with Streamlit, TensorFlow, and librosa | 
        <a href="https://github.com/yourproject/respiratory-disease-detection">View on GitHub</a></p>
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    app = RespiratoryDiseaseApp()
    app.run()