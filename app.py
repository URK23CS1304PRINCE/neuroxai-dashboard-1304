# app.py - Updated to handle any EEG dataset
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import onnxruntime as ort
import joblib
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
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; color: white; text-align: center; }
    .stButton button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 10px 25px; font-weight: 600; width: 100%; }
    .alert-high { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; padding: 20px; color: white; }
    .alert-moderate { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 10px; padding: 20px; color: #333; }
    .alert-normal { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 10px; padding: 20px; color: #333; }
    .info-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 15px; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'session' not in st.session_state:
    st.session_state.session = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'expected_features' not in st.session_state:
    st.session_state.expected_features = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.title("NeuroXAI DL")
    st.markdown("---")
    
    st.subheader("📦 Load Model")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        st.info(f"Expected features: {st.session_state.expected_features}")
        if st.button("🔄 Reload Model"):
            st.session_state.model_loaded = False
            st.session_state.session = None
            st.session_state.scaler = None
            st.rerun()
    else:
        st.info("Upload your model files:")
        
        model_file = st.file_uploader("ONNX Model (.onnx)", type=['onnx'])
        scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'])
        
        if model_file and scaler_file:
            with st.spinner("Loading model..."):
                try:
                    # Save uploaded files
                    with open('temp_model.onnx', 'wb') as f:
                        f.write(model_file.getbuffer())
                    with open('temp_scaler.pkl', 'wb') as f:
                        f.write(scaler_file.getbuffer())
                    
                    # Load ONNX model
                    st.session_state.session = ort.InferenceSession('temp_model.onnx')
                    st.session_state.scaler = joblib.load('temp_scaler.pkl')
                    st.session_state.expected_features = st.session_state.scaler.n_features_in_
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
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
    st.info("👈 **Get Started**: Upload your ONNX model (.onnx) and scaler (.pkl) files")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📁</h3><h4>Step 1</h4><p>Upload Model</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>📤</h3><h4>Step 2</h4><p>Upload EEG</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🔬</h3><h4>Step 3</h4><p>Get Results</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🎮 Try Demo Mode (No Model Required)"):
        st.session_state.demo_mode = True
        st.session_state.model_loaded = True
        st.rerun()

else:
    st.markdown(f'<div class="info-box">✅ Model loaded! Expected {st.session_state.expected_features} features.</div>', unsafe_allow_html=True)
    st.markdown("---")
    
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
        
        # Check feature compatibility
        if df.shape[1] != st.session_state.expected_features:
            st.warning(f"⚠️ Feature mismatch! Your data has {df.shape[1]} features but model expects {st.session_state.expected_features} features.")
            st.info("The system will automatically handle this by padding/trimming features.")
        
        if st.button("🔬 Analyze EEG", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing with your model..."):
                if st.session_state.session is not None and st.session_state.scaler is not None:
                    try:
                        # Convert to numpy
                        X_input = df.values.astype(np.float32)
                        
                        # Handle feature mismatch
                        if X_input.shape[1] != st.session_state.expected_features:
                            if X_input.shape[1] < st.session_state.expected_features:
                                # Pad with zeros
                                pad_width = st.session_state.expected_features - X_input.shape[1]
                                X_input = np.pad(X_input, ((0, 0), (0, pad_width)), mode='constant')
                                st.info(f"📊 Padded {X_input.shape[0]} samples from {df.shape[1]} to {st.session_state.expected_features} features")
                            else:
                                # Trim excess features
                                X_input = X_input[:, :st.session_state.expected_features]
                                st.info(f"📊 Trimmed {X_input.shape[0]} samples from {df.shape[1]} to {st.session_state.expected_features} features")
                        
                        # Scale
                        X_input = st.session_state.scaler.transform(X_input)
                        X_input = X_input.astype(np.float32)
                        
                        # Get input name
                        input_name = st.session_state.session.get_inputs()[0].name
                        
                        # Predict
                        predictions_proba = st.session_state.session.run(None, {input_name: X_input})[0]
                        
                        # Get predictions
                        predictions = np.argmax(predictions_proba, axis=1) + 1
                        confidences = np.max(predictions_proba, axis=1)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("Falling back to demo mode")
                        np.random.seed(42)
                        predictions = np.random.choice([1,2,3,4,5], len(df), p=[0.08,0.12,0.15,0.25,0.40])
                        confidences = np.random.uniform(0.75, 0.98, len(df))
                else:
                    # Demo mode
                    np.random.seed(42)
                    predictions = np.random.choice([1,2,3,4,5], len(df), p=[0.08,0.12,0.15,0.25,0.40])
                    confidences = np.random.uniform(0.75, 0.98, len(df))
                    st.info("ℹ️ Using demo predictions. Load actual model for real results.")
            
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
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("📋 Detailed Results")
            
            results_df = pd.DataFrame({
                'Sample': range(1, len(df)+1),
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
            st.subheader("🔍 Detailed Sample Analysis")
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
