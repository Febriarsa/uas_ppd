import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
@st.cache_resource
def load_model():
    model_path = os.path.join(SCRIPT_DIR, 'best_rf_model.joblib')
    scaler_path = os.path.join(SCRIPT_DIR, 'scaler.joblib')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Page configuration
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e88e5 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(30, 136, 229, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.3rem;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.75rem;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #4fc3f7;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .potable {
        background: linear-gradient(135deg, #00c853 0%, #69f0ae 100%);
        border: 2px solid #00c853;
    }
    
    .not-potable {
        background: linear-gradient(135deg, #ff1744 0%, #ff8a80 100%);
        border: 2px solid #ff1744;
    }
    
    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .result-text {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    .result-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* History section */
    .history-card {
        background: linear-gradient(135deg, #263238 0%, #37474f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1e88e5;
    }
    
    .history-potable {
        border-left-color: #00c853;
    }
    
    .history-not-potable {
        border-left-color: #ff1744;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #263238 0%, #37474f 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #1e88e5;
    }
    
    .info-card h4 {
        color: #4fc3f7;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .info-card p {
        color: rgba(255,255,255,0.8);
        margin: 0;
        font-size: 0.85rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 30px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 136, 229, 0.6);
    }
    
    /* Section headers */
    .section-header {
        color: #4fc3f7;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(79, 195, 247, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💧 Water Potability Predictor</h1>
    <p>Prediksi Kelayakan Air Minum Menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("## 📊 Parameter Kualitas Air")
st.sidebar.markdown("Masukkan nilai parameter air yang ingin diprediksi:")

# Feature inputs with descriptions and icons
features = {
    'ph': {
        'label': 'pH Level',
        'min': 0.0, 'max': 14.0, 'default': 7.0,
        'help': 'Tingkat keasaman air (0-14). Air minum ideal: 6.5-8.5',
        'icon': '🧪'
    },
    'Hardness': {
        'label': 'Hardness (mg/L)',
        'min': 47.0, 'max': 324.0, 'default': 180.0,
        'help': 'Kesadahan air, diukur dari kandungan kalsium dan magnesium',
        'icon': '💎'
    },
    'Solids': {
        'label': 'Total Dissolved Solids (ppm)',
        'min': 320.0, 'max': 61230.0, 'default': 20000.0,
        'help': 'Total padatan terlarut dalam air',
        'icon': '🔬'
    },
    'Chloramines': {
        'label': 'Chloramines (ppm)',
        'min': 0.35, 'max': 13.13, 'default': 6.0,
        'help': 'Kadar kloramin sebagai disinfektan',
        'icon': '🧴'
    },
    'Sulfate': {
        'label': 'Sulfate (mg/L)',
        'min': 129.0, 'max': 481.0, 'default': 300.0,
        'help': 'Kandungan sulfat dalam air',
        'icon': '⚗️'
    },
    'Conductivity': {
        'label': 'Conductivity (μS/cm)',
        'min': 181.0, 'max': 753.0, 'default': 400.0,
        'help': 'Kemampuan air menghantarkan listrik',
        'icon': '⚡'
    },
    'Organic_carbon': {
        'label': 'Organic Carbon (ppm)',
        'min': 2.2, 'max': 28.3, 'default': 12.0,
        'help': 'Kandungan karbon organik dalam air',
        'icon': '🌿'
    },
    'Trihalomethanes': {
        'label': 'Trihalomethanes (μg/L)',
        'min': 0.74, 'max': 124.0, 'default': 70.0,
        'help': 'Senyawa kimia hasil reaksi klorin dengan bahan organik',
        'icon': '☢️'
    },
    'Turbidity': {
        'label': 'Turbidity (NTU)',
        'min': 1.45, 'max': 6.74, 'default': 3.0,
        'help': 'Tingkat kekeruhan air',
        'icon': '🌫️'
    }
}

# Collect input values
input_data = {}
for key, config in features.items():
    input_data[key] = st.sidebar.slider(
        config['label'],
        min_value=config['min'],
        max_value=config['max'],
        value=config['default'],
        help=config['help']
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">📋 Nilai Parameter yang Dimasukkan</div>', unsafe_allow_html=True)
    
    # Display input values as metric cards (3x3 grid)
    feature_keys = list(features.keys())
    
    # Row 1
    cols_row1 = st.columns(3)
    for i, col in enumerate(cols_row1):
        key = feature_keys[i]
        config = features[key]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{config['icon']}</div>
                <div class="metric-label">{config['label'].split('(')[0].strip()}</div>
                <div class="metric-value">{input_data[key]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Row 2
    cols_row2 = st.columns(3)
    for i, col in enumerate(cols_row2):
        key = feature_keys[i + 3]
        config = features[key]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{config['icon']}</div>
                <div class="metric-label">{config['label'].split('(')[0].strip()}</div>
                <div class="metric-value">{input_data[key]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Row 3
    cols_row3 = st.columns(3)
    for i, col in enumerate(cols_row3):
        key = feature_keys[i + 6]
        config = features[key]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{config['icon']}</div>
                <div class="metric-label">{config['label'].split('(')[0].strip()}</div>
                <div class="metric-value">{input_data[key]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction History Section
    st.markdown('<div class="section-header">📜 Riwayat Prediksi</div>', unsafe_allow_html=True)
    
    if len(st.session_state.prediction_history) == 0:
        st.info("Belum ada riwayat prediksi. Klik tombol 'Analisis Kelayakan Air' untuk memulai.")
    else:
        # Show history in reverse order (newest first)
        for i, record in enumerate(reversed(st.session_state.prediction_history[-5:])):  # Show last 5
            result_class = "history-potable" if record['result'] == 1 else "history-not-potable"
            result_text = "✅ Layak" if record['result'] == 1 else "⚠️ Tidak Layak"
            st.markdown(f"""
            <div class="history-card {result_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #4fc3f7;">#{len(st.session_state.prediction_history) - i}</strong>
                        <span style="color: rgba(255,255,255,0.6); margin-left: 10px; font-size: 0.8rem;">{record['time']}</span>
                    </div>
                    <div>
                        <span style="font-weight: bold;">{result_text}</span>
                        <span style="color: rgba(255,255,255,0.7); margin-left: 10px;">({record['confidence']:.1f}%)</span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.75rem; color: rgba(255,255,255,0.5);">
                    pH: {record['params']['ph']:.1f} | Hardness: {record['params']['Hardness']:.0f} | TDS: {record['params']['Solids']:.0f} | Turb: {record['params']['Turbidity']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear history button
        if st.button("🗑️ Hapus Riwayat", key="clear_history"):
            st.session_state.prediction_history = []
            st.rerun()

with col2:
    st.markdown('<div class="section-header">🔬 Prediksi</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Analisis Kelayakan Air", use_container_width=True):
        # Prepare input for prediction
        input_array = pd.DataFrame([input_data])
        expected_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                           'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        input_array = input_array[expected_columns]
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Save to history
        confidence = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
        st.session_state.prediction_history.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'result': int(prediction),
            'confidence': confidence,
            'params': input_data.copy()
        })
        
        # Display result
        if prediction == 1:
            st.markdown("""
            <div class="result-card potable">
                <div class="result-icon">✅</div>
                <div class="result-text">LAYAK MINUM</div>
                <div class="result-subtitle">Air ini aman untuk dikonsumsi</div>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"Confidence: {prediction_proba[1]*100:.1f}%")
        else:
            st.markdown("""
            <div class="result-card not-potable">
                <div class="result-icon">⚠️</div>
                <div class="result-text">TIDAK LAYAK MINUM</div>
                <div class="result-subtitle">Air ini tidak aman untuk dikonsumsi</div>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"Confidence: {prediction_proba[0]*100:.1f}%")
        
        # Show probability breakdown
        st.markdown("#### 📊 Probabilitas")
        prob_df = pd.DataFrame({
            'Kategori': ['Tidak Layak', 'Layak'],
            'Probabilitas': [prediction_proba[0], prediction_proba[1]]
        })
        st.bar_chart(prob_df.set_index('Kategori'))

# Information section
st.markdown("---")
st.markdown("### ℹ️ Informasi Parameter Kualitas Air")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="info-card">
        <h4>🧪 pH Level</h4>
        <p>Mengukur tingkat keasaman atau kebasaan air. Nilai ideal: 6.5 - 8.5</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>💎 Hardness</h4>
        <p>Kandungan mineral (kalsium & magnesium) dalam air.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🔬 TDS (Solids)</h4>
        <p>Total padatan terlarut dalam air.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="info-card">
        <h4>🧴 Chloramines</h4>
        <p>Disinfektan untuk membunuh bakteri dalam air.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>⚗️ Sulfate</h4>
        <p>Senyawa sulfat alami dalam air.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>⚡ Conductivity</h4>
        <p>Kemampuan air menghantarkan listrik.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div class="info-card">
        <h4>🌿 Organic Carbon</h4>
        <p>Kandungan bahan organik dalam air.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>☢️ Trihalomethanes</h4>
        <p>Senyawa hasil reaksi klorin dengan bahan organik.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🌫️ Turbidity</h4>
        <p>Tingkat kekeruhan air.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🎓 <strong>UAS Praktikum Penambangan Data</strong> | Universitas Gadjah Mada</p>
    <p>Model: Random Forest Classifier | Akurasi: ~65%</p>
</div>
""", unsafe_allow_html=True)
