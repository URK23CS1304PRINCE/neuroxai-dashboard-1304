# app.py - Final Working Version for Python 3.14
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
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
        width: 100%;
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        background: #f8f9fa;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.title("NeuroXAI DL")
    st.markdown("---")
    
    st.subheader("📦 Load Model")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        if st.button("🔄 Reload Model"):
            st.session_state.model_loaded = False
            st.rerun()
    else:
        st.info("Upload your trained model files:")
        
        model_file = st.file_uploader("Model (.keras or .h5)", type=['keras', 'h5'])
        scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'])
        
        if model_file and scaler_file:
            with st.spinner("Loading model..."):
                try:
                    # Try to load with tensorflow if available
                    try:
                        import tensorflow as tf
                        from tensorflow import keras
                        
                        with open('temp_model.keras', 'wb') as f:
                            f.write(model_file.getbuffer())
                        with open('temp_scaler.pkl', 'wb') as f:
                            f.write(scaler_file.getbuffer())
                        
                        st.session_state.model = keras.models.load_model('temp_model.keras')
                        st.session_state.scaler = joblib.load('temp_scaler.pkl')
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    except ImportError:
                        st.warning("⚠️ TensorFlow not available. Using demo mode.")
                        st.session_state.demo_mode = True
                        st.session_state.model_loaded = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.caption("© 2024 NeuroXAI DL")

# Main content
st.title("🧠 NeuroXAI DL")
st.markdown("### Advanced EEG-based Seizure Detection System")
st.markdown("---")

if not st.session_state.model_loaded:
    st.info("👈 **Get Started**: Load your trained model files in the sidebar")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁</h3>
            <h4>Step 1</h4>
            <p>Upload Model</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📤</h3>
            <h4>Step 2</h4>
            <p>Upload EEG Data</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬</h3>
            <h4>Step 3</h4>
            <p>Get Predictions</p>
        </div>
        """, unsafe_allow_html=True)
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
                if st.session_state.model is not None:
                    predictions_proba = st.session_state.model.predict(X_input, verbose=0)
                    predictions = np.argmax(predictions_proba, axis=1) + 1
                    confidences = np.max(predictions_proba, axis=1)
                else:
                    # Demo predictions
                    np.random.seed(42)
                    predictions = np.random.choice([1,2,3,4,5], len(df), p=[0.08,0.12,0.15,0.25,0.40])
                    confidences = np.random.uniform(0.75, 0.98, len(df))
                    st.info("ℹ️ Using demo mode. Load actual model for real predictions.")
            
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
            
            # Results table
            st.subheader("📋 Detailed Results")
            
            results_df = pd.DataFrame({
                'Sample': range(1, len(df)+1),
                'Predicted Class': predictions,
                'Risk Level': ['🔴 High' if p<=2 else '🟡 Borderline' if p==3 else '🟢 Low' if p==4 else '✅ Normal' for p in predictions],
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
                        <p><strong>Action Required:</strong> Immediate neurological consultation</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_class == 2:
                    st.markdown(f"""
                    <div class="alert-moderate">
                        <h3>🟠 MODERATE RISK - ABNORMAL ACTIVITY</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Action Required:</strong> Follow-up within 1 week</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_class == 3:
                    st.markdown(f"""
                    <div class="alert-moderate">
                        <h3>🟡 BORDERLINE - SUBTLE ABNORMALITIES</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Action Required:</strong> Repeat EEG in 2-4 weeks</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-normal">
                        <h3>✅ LOW RISK / NORMAL</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Action Required:</strong> Routine monitoring</p>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center;'>🧠 NeuroXAI DL | AI-Powered EEG Analysis | Clinical Decision Support</p>", unsafe_allow_html=True)
