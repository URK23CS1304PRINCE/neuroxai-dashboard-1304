
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroXAI DL - Seizure Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .alert-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .alert-moderate {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 10px;
        padding: 20px;
        color: #333;
    }
    .alert-normal {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 20px;
        color: #333;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.title("NeuroXAI DL")
    st.markdown("---")
    
    # Model Loading Section
    st.subheader("📦 Model Loading")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded Successfully!")
        st.info(f"Model: Neural Network
Classes: 5
Status: Ready")
    else:
        st.warning("⚠️ Model Not Loaded")
        st.markdown("Please upload your trained model files:")
        
        model_file = st.file_uploader("Model (.keras)", type=['keras'], key="model")
        scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'], key="scaler")
        label_file = st.file_uploader("Label Map (.pkl)", type=['pkl'], key="label")
        inverse_file = st.file_uploader("Inverse Map (.pkl)", type=['pkl'], key="inverse")
        
        if model_file and scaler_file and label_file and inverse_file:
            with st.spinner("Loading model files..."):
                try:
                    # Save uploaded files temporarily
                    with open('temp_model.keras', 'wb') as f:
                        f.write(model_file.getbuffer())
                    with open('temp_scaler.pkl', 'wb') as f:
                        f.write(scaler_file.getbuffer())
                    with open('temp_label.pkl', 'wb') as f:
                        f.write(label_file.getbuffer())
                    with open('temp_inverse.pkl', 'wb') as f:
                        f.write(inverse_file.getbuffer())
                    
                    # Load model and objects
                    st.session_state.model = keras.models.load_model('temp_model.keras')
                    st.session_state.scaler = joblib.load('temp_scaler.pkl')
                    st.session_state.label_map = joblib.load('temp_label.pkl')
                    st.session_state.inverse_map = joblib.load('temp_inverse.pkl')
                    st.session_state.model_loaded = True
                    
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    st.markdown("---")
    st.caption("© 2024 NeuroXAI DL")
    st.caption("Powered by TensorFlow")

# Main content
st.title("🧠 NeuroXAI DL")
st.markdown("### Advanced EEG-based Seizure Detection System")
st.markdown("---")

if not st.session_state.model_loaded:
    st.info("👈 **Get Started**: Load your trained model files using the sidebar")
    st.markdown("""
    ### How to use:
    1. **Upload your model files** in the sidebar
    2. **Upload EEG data** below
    3. **Get instant predictions** from your trained model
    
    ### Files needed:
    - `neuroxai_trained_model.keras` - Your trained neural network
    - `eeg_scaler.pkl` - Data preprocessing scaler
    - `label_map.pkl` - Class label mapping
    - `inverse_map.pkl` - Reverse class mapping
    """)
else:
    # Show model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Status", "✅ Active")
    with col2:
        st.metric("Model Type", "Dense Neural Network")
    with col3:
        st.metric("Classes", "5 (Risk Levels)")
    
    st.markdown("---")
    
    # File upload for EEG data
    st.subheader("📤 Upload EEG Data for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload EEG data in CSV format"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Clean data
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'y' in df.columns:
            df = df.drop('y', axis=1)
        
        st.success(f"✅ Loaded {len(df)} samples with {df.shape[1]} features")
        
        with st.expander("📊 Preview Data"):
            st.dataframe(df.head())
        
        # Analysis button
        if st.button("🔬 Analyze EEG", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing with your trained model..."):
                # Preprocess
                X_input = st.session_state.scaler.transform(df)
                
                # Predict
                predictions_proba = st.session_state.model.predict(X_input, verbose=0)
                predictions_mapped = np.argmax(predictions_proba, axis=1)
                predictions = np.array([st.session_state.inverse_map[p] for p in predictions_mapped])
                confidences = np.max(predictions_proba, axis=1)
                
                st.session_state.predictions = predictions
                st.session_state.confidences = confidences
                st.session_state.predictions_made = True
            
            st.success("✅ Analysis complete!")
            
            # Summary metrics
            st.subheader("📊 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            high_risk = np.sum(predictions <= 2)
            moderate_risk = np.sum(predictions == 3)
            low_risk = np.sum(predictions == 4)
            normal = np.sum(predictions == 5)
            
            with col1:
                st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/len(predictions)*100:.0f}%")
            with col2:
                st.metric("🟡 Borderline", moderate_risk, delta=f"{moderate_risk/len(predictions)*100:.0f}%")
            with col3:
                st.metric("🟢 Low Risk", low_risk, delta=f"{low_risk/len(predictions)*100:.0f}%")
            with col4:
                st.metric("✅ Normal", normal, delta=f"{normal/len(predictions)*100:.0f}%")
            
            # Risk distribution chart
            risk_data = pd.DataFrame({
                'Risk Level': ['High Risk', 'Borderline', 'Low Risk', 'Normal'],
                'Count': [high_risk, moderate_risk, low_risk, normal]
            })
            fig = px.bar(risk_data, x='Risk Level', y='Count', color='Risk Level',
                         color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'],
                         title="Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Predictions")
            
            results_df = pd.DataFrame({
                'Sample': range(1, len(predictions) + 1),
                'Predicted Class': predictions,
                'Risk Level': ['🔴 High Risk' if p<=2 else '🟡 Borderline' if p==3 else '🟢 Low Risk' if p==4 else '✅ Normal' for p in predictions],
                'Confidence': [f"{c:.2%}" for c in confidences]
            })
            
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="neuroxai_predictions.csv",
                mime="text/csv"
            )
            
            # Sample analysis
            st.subheader("🔍 Sample Analysis")
            sample_idx = st.number_input("Select Sample Number", min_value=1, max_value=len(predictions), value=1)
            
            if sample_idx:
                idx = sample_idx - 1
                pred_class = predictions[idx]
                confidence = confidences[idx]
                
                if pred_class == 1:
                    st.markdown(f"""
                    <div class="alert-high">
                        <h3>🔴 HIGH RISK - SEIZURE ACTIVITY DETECTED</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Clinical Action Required:</strong></p>
                        <ul>
                            <li>Immediate neurological consultation</li>
                            <li>Schedule follow-up EEG within 24-48 hours</li>
                            <li>Consider anti-epileptic medication</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_class == 2:
                    st.markdown(f"""
                    <div class="alert-moderate">
                        <h3>🟠 MODERATE RISK - ABNORMAL ACTIVITY</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Clinical Action Required:</strong></p>
                        <ul>
                            <li>Neurological follow-up within 1 week</li>
                            <li>Consider sleep-deprived EEG</li>
                            <li>Monitor for clinical symptoms</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_class == 3:
                    st.markdown(f"""
                    <div class="alert-moderate">
                        <h3>🟡 BORDERLINE - SUBTLE ABNORMALITIES</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Clinical Action Required:</strong></p>
                        <ul>
                            <li>Repeat EEG in 2-4 weeks</li>
                            <li>Clinical correlation advised</li>
                            <li>Consider ambulatory EEG monitoring</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-normal">
                        <h3>✅ LOW RISK / NORMAL</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Clinical Action Required:</strong></p>
                        <ul>
                            <li>Routine follow-up as clinically indicated</li>
                            <li>Patient reassurance</li>
                            <li>Return to normal activities</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>🧠 NeuroXAI DL | Powered by Your Trained Model | Clinical Decision Support</p>",
    unsafe_allow_html=True
)
