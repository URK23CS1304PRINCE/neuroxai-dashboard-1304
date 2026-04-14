# app.py - Updated version that works with Python 3.14
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import h5py
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroXAI DL - Seizure Detection",
    page_icon="🧠",
    layout="wide"
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
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
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

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.title("NeuroXAI DL")
    st.markdown("---")
    
    st.subheader("📦 Load Model")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
    else:
        st.info("Upload your trained model files:")
        
        model_file = st.file_uploader("Model (.keras)", type=['keras', 'h5'])
        scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'])
        
        if model_file and scaler_file:
            with st.spinner("Loading model..."):
                try:
                    # Save files
                    with open('temp_model.keras', 'wb') as f:
                        f.write(model_file.getbuffer())
                    with open('temp_scaler.pkl', 'wb') as f:
                        f.write(scaler_file.getbuffer())
                    
                    # Try to import tensorflow only when needed
                    try:
                        import tensorflow as tf
                        from tensorflow import keras
                        st.session_state.model = keras.models.load_model('temp_model.keras')
                        st.session_state.scaler = joblib.load('temp_scaler.pkl')
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    except ImportError:
                        st.error("⚠️ TensorFlow not available. Using demo mode.")
                        st.session_state.demo_mode = True
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.caption("© 2024 NeuroXAI DL")

# Main content
st.title("🧠 NeuroXAI DL")
st.markdown("### Advanced EEG-based Seizure Detection")
st.markdown("---")

if not st.session_state.model_loaded:
    st.info("👈 **Get Started**: Load your trained model files in the sidebar")
    st.markdown("""
    ### How to use:
    1. **Upload your model files** in the sidebar
    2. **Upload EEG data** below
    3. **Get predictions** from your trained model
    
    ### Files needed:
    - `neuroxai_trained_model.keras` - Your trained neural network
    - `eeg_scaler.pkl` - Data preprocessing scaler
    """)
else:
    # File upload for EEG data
    st.subheader("📤 Upload EEG Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Clean data
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'y' in df.columns:
            df = df.drop('y', axis=1)
        
        st.success(f"✅ Loaded {len(df)} samples with {df.shape[1]} features")
        
        with st.expander("📊 Preview Data"):
            st.dataframe(df.head())
        
        if st.button("🔬 Analyze EEG", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing with your model..."):
                # Preprocess
                X_input = st.session_state.scaler.transform(df)
                
                # Predict
                if hasattr(st.session_state, 'model'):
                    predictions_proba = st.session_state.model.predict(X_input, verbose=0)
                    predictions = np.argmax(predictions_proba, axis=1) + 1
                    confidences = np.max(predictions_proba, axis=1)
                else:
                    # Demo predictions
                    np.random.seed(42)
                    predictions = np.random.choice([1,2,3,4,5], len(df), p=[0.08,0.12,0.15,0.25,0.40])
                    confidences = np.random.uniform(0.75, 0.98, len(df))
            
            st.success("✅ Analysis complete!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            high_risk = np.sum(predictions <= 2)
            normal = np.sum(predictions == 5)
            
            with col1:
                st.metric("🔴 High Risk", high_risk)
            with col2:
                st.metric("🟡 Borderline", np.sum(predictions == 3))
            with col3:
                st.metric("🟢 Low Risk", np.sum(predictions == 4))
            with col4:
                st.metric("✅ Normal", normal)
            
            # Results table
            results_df = pd.DataFrame({
                'Sample': range(1, len(df)+1),
                'Predicted Class': predictions,
                'Confidence': [f"{c:.2%}" for c in confidences]
            })
            st.dataframe(results_df, use_container_width=True)
            
            # Download
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "predictions.csv")

st.markdown("---")
st.markdown("<p style='text-align: center;'>🧠 NeuroXAI DL | AI-Powered EEG Analysis</p>", unsafe_allow_html=True)
