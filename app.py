# app.py - Complete Working Version (No duplicate keys)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
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
    .success-box { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); border-radius: 10px; padding: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'weights_data' not in st.session_state:
    st.session_state.weights_data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Neural Network class using NumPy only
class SimpleNeuralNetwork:
    def __init__(self, weights_data):
        self.layer_weights = []
        self.layer_biases = []
        self._build_from_weights(weights_data)
    
    def _build_from_weights(self, weights_data):
        """Build network from extracted weights"""
        layer_weights = {}
        layer_biases = {}
        
        for key, value in weights_data.items():
            if 'weights' in key:
                parts = key.split('_')
                if len(parts) >= 3:
                    layer_num = parts[2]
                else:
                    layer_num = parts[1]
                layer_weights[layer_num] = np.array(value)
            elif 'bias' in key:
                parts = key.split('_')
                if len(parts) >= 3:
                    layer_num = parts[2]
                else:
                    layer_num = parts[1]
                layer_biases[layer_num] = np.array(value)
        
        # Sort layers
        sorted_keys = sorted(layer_weights.keys(), key=lambda x: int(x))
        self.layer_weights = [layer_weights[k] for k in sorted_keys]
        self.layer_biases = [layer_biases[k] for k in sorted_keys]
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def predict(self, X):
        """Forward pass through the network"""
        x = X
        for i in range(len(self.layer_weights) - 1):
            x = self.relu(np.dot(x, self.layer_weights[i]) + self.layer_biases[i])
        
        # Last layer
        x = np.dot(x, self.layer_weights[-1]) + self.layer_biases[-1]
        return self.softmax(x)

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
            st.session_state.weights_data = None
            st.session_state.scaler = None
            st.rerun()
    else:
        st.info("Upload your model files:")
        
        weights_file = st.file_uploader("Model Weights (.json)", type=['json'])
        scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'])
        
        if weights_file and scaler_file:
            with st.spinner("Loading model..."):
                try:
                    # Load weights
                    st.session_state.weights_data = json.load(weights_file)
                    # Load scaler
                    st.session_state.scaler = joblib.load(scaler_file)
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
    st.info("👈 **Get Started**: Upload your model weights (.json) and scaler (.pkl) files")
    
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
    st.markdown('<div class="success-box">✅ Model loaded successfully! Ready for analysis.</div>', unsafe_allow_html=True)
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
        
        if st.button("🔬 Analyze EEG", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing with your neural network..."):
                # Check if we have real model or demo mode
                if st.session_state.weights_data is not None and st.session_state.scaler is not None:
                    # Use real model
                    model = SimpleNeuralNetwork(st.session_state.weights_data)
                    X_input = st.session_state.scaler.transform(df)
                    predictions_proba = model.predict(X_input)
                    predictions = np.argmax(predictions_proba, axis=1) + 1
                    confidences = np.max(predictions_proba, axis=1)
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
                st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/len(predictions)*100:.0f}%", delta_color="inverse")
            with col2:
                st.metric("🟡 Borderline", moderate_risk, delta=f"{moderate_risk/len(predictions)*100:.0f}%")
            with col3:
                st.metric("🟢 Low Risk", low_risk, delta=f"{low_risk/len(predictions)*100:.0f}%")
            with col4:
                st.metric("✅ Normal", normal, delta=f"{normal/len(predictions)*100:.0f}%")
            
            # Risk distribution chart
            risk_data = pd.DataFrame({
                'Risk Level': ['High Risk', 'Borderline', 'Low Risk', 'Normal'],
                'Count': [high_risk, moderate_risk, low_risk, normal],
                'Color': ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
            })
            
            fig = px.bar(risk_data, x='Risk Level', y='Count', color='Risk Level',
                         color_discrete_sequence=risk_data['Color'],
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

st.markdown("---")
st.markdown("<p style='text-align: center;'>🧠 NeuroXAI DL | AI-Powered EEG Analysis | Clinical Decision Support System</p>", unsafe_allow_html=True)
