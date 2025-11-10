

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BACKGROUND_URL = "https://img.freepik.com/premium-photo/blue-dots-background-with-stylish-dots_961875-93311.jpg "
MODEL_PATH = "knn_model_ws.joblib"

# Features - EXACT FORMAT FROM YOUR TRAINING DATA
ALL_FEATURES = [
    'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend',
    'Last Interaction', 'Support_Intensity', 'Avg_Spend_per_Month', 'Customer_Value'
]

# Transparent encoding maps - MATCH FEATURE NAMES EXACTLY
ENCODING_MAPS = {
    'Gender': {'Female': 0, 'Male': 1},
    'Subscription Type': {'Basic': 0, 'Premium': 1, 'Standard': 2},
    'Contract Length': {'Annual': 0, 'Monthly': 1, 'Quarterly': 2}
}

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="AI Customer Churn Prediction System",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Dark theme with glossy turquoise sky blue title
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("{BACKGROUND_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* --- GLOSSY TURQUOISE SKY BLUE TITLE --- */
    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;  /* Pure white base */
        text-shadow: 0 0 10px #ffffff, 0 0 20px #00ffff, 0 0 30px #40e0d0, 0 0 40px #00ced1;
        animation: turquoise-glow 2s ease-in-out infinite alternate;
        padding: 2rem 0 1rem 0;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }}
    @keyframes turquoise-glow {{
        from {{ text-shadow: 0 0 10px #ffffff, 0 0 20px #00ffff, 0 0 30px #40e0d0, 0 0 40px #00ced1; }}
        to {{ text-shadow: 0 0 15px #ffffff, 0 0 30px #7fffd4, 0 0 45px #48d1cc, 0 0 60px #00ced1; }}
    }}
    
    /* --- SHINY GOLD HEADER --- */
    .gold-header {{
        color: #FFD700;  /* Gold */
        font-size: 1.5rem;
        font-weight: 700;
        text-shadow: 0 0 10px #FFD700, 0 0 20px #FFA500, 0 0 30px #FF8C00;
        animation: gold-glow 1.5s ease-in-out infinite alternate;
        margin: 1rem 0;
        padding: 0.5rem 0;
    }}
    @keyframes gold-glow {{
        from {{ text-shadow: 0 0 10px #FFD700, 0 0 20px #FFA500, 0 0 30px #FF8C00; }}
        to {{ text-shadow: 0 0 15px #FFD700, 0 0 30px #FFA500, 0 0 40px #FF8C00, 0 0 50px #FF6347; }}
    }}
    
    .survey-container {{
        background: rgba(0, 0, 0, 0.8);
        border-radius: 15px;
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 204, 255, 0.3);
        backdrop-filter: blur(10px);
    }}
    
    .section-header {{
        color: #00ccff;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #00ccff;
        padding-bottom: 0.5rem;
    }}
    
    .question-label {{
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
    }}
    
    .submit-button {{
        background: linear-gradient(135deg, #00ccff 0%, #0099ff 100%);
        color: white;
        font-weight: 800;
        font-size: 1.3rem;
        padding: 1rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 204, 255, 0.4);
        width: 100%;
        margin-top: 2rem;
    }}
    
    .results-card {{
        background: rgba(0, 0, 0, 0.9);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(0, 204, 255, 0.5);
    }}
    
    .metric-box {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 0.5rem;
    }}
    
    .recommendation-box {{
        background: rgba(30, 60, 114, 0.7);
        border-left: 4px solid #00ccff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: white;
    }}
    
    .st-emotion-cache-1y4p8pa {{
        background: rgba(0, 0, 0, 0.3) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== MODEL & ENCODING ====================
@st.cache_resource(show_spinner="🚀 Loading AI Model...")
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

def encode_categoricals(data: dict):
    """Transparent encoding"""
    encoded = data.copy()
    for feature, mapping in ENCODING_MAPS.items():
        if feature in encoded:
            encoded[feature] = mapping.get(encoded[feature], -1)
    return encoded

def preprocess_input(input_data: dict):
    encoded_data = encode_categoricals(input_data)
    return pd.DataFrame([encoded_data])[ALL_FEATURES]

# ==================== VISUALIZATIONS ====================
def create_gauge_chart(probability: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)", 'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': 'white'}},
            'bar': {'color': "#00ccff", 'thickness': 0.4},
            'bgcolor': "rgba(0,0,0,0.3)",
            'bordercolor': "#00ccff",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.7)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.7)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.7)'}
            ]
        }
    ))
    fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

# ==================== RECOMMENDATIONS ====================
def generate_recommendations(prob: float, data: dict):
    recs = []
    risk = "High" if prob >= 70 else "Medium" if prob >= 40 else "Low"
    
    if risk == "High":
        recs.append(("🚨 CRITICAL", f"Contact within 24h! Risk: {prob}%"))
    elif risk == "Medium":
        recs.append(("⚠️ PRIORITY", "Schedule check-in within 7 days"))
    else:
        recs.append(("✅ STABLE", "Focus on upsell opportunities"))
    
    # MATCHING YOUR EXACT COLUMN FORMAT
    if data.get('Support Calls', 0) > 5:
        recs.append(("📞 SUPPORT", f"{data['Support Calls']} calls - review support quality"))
    if data.get('Payment Delay', 0) > 10:
        recs.append(("💳 PAYMENT", f"{data['Payment Delay']} days delay - offer payment plan"))
    if data.get('Usage Frequency', 0) < 10:
        recs.append(("📈 ADOPTION", f"Low usage ({data['Usage Frequency']}) - provide training"))
    
    return recs

# ==================== MAIN APP ====================
def main():
    # --- GLOSSY TURQUOISE SKY BLUE TITLE ---
    st.markdown('<h1 class="main-title">AI Customer Churn Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Load model (silent)
    model = load_model()
    
    # --- SURVEY FORM ---
    st.markdown('<div class="survey-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📋 Customer Demographics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="question-label">Customer Age</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Gender</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Tenure (months)</div>', unsafe_allow_html=True)
        tenure = st.number_input("Tenure", min_value=0, max_value=120, value=24, step=1, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Usage Frequency (per month)</div>', unsafe_allow_html=True)
        usage_freq = st.number_input("Usage_Frequency", min_value=0, max_value=100, value=15, step=1, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Support Calls</div>', unsafe_allow_html=True)
        support_calls = st.number_input("Support_Calls", min_value=0, max_value=50, value=2, step=1, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Payment Delay (days)</div>', unsafe_allow_html=True)
        payment_delay = st.number_input("Payment_Delay", min_value=0, max_value=365, value=0, step=1, label_visibility="collapsed")
    
    with col2:
        st.markdown('<div class="question-label">Subscription Type</div>', unsafe_allow_html=True)
        subscription = st.selectbox("Subscription_Type", ["Basic", "Standard", "Premium"], label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Contract Length</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract_Length", ["Annual", "Monthly", "Quarterly"], label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Total Spend ($)</div>', unsafe_allow_html=True)
        total_spend = st.number_input("Total_Spend", min_value=0.0, value=2400.0, step=0.01, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Last Interaction (days ago)</div>', unsafe_allow_html=True)
        last_interaction = st.number_input("Last_Interaction", min_value=0, max_value=365, value=5, step=1, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Support Intensity (0-1)</div>', unsafe_allow_html=True)
        support_intensity = st.slider("Support_Intensity", 0.0, 1.0, 0.1, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Avg Spend per Month ($)</div>', unsafe_allow_html=True)
        avg_spend = st.number_input("Avg_Spend_per_Month", min_value=0.0, value=100.0, step=0.01, label_visibility="collapsed")
        
        st.markdown('<div class="question-label">Customer Value ($)</div>', unsafe_allow_html=True)
        customer_value = st.number_input("Customer_Value", min_value=0.0, value=2500.0, step=0.01, label_visibility="collapsed")
    
    if st.button("🔮 **GET CHURN PREDICTION**", type="primary", use_container_width=True, key="predict"):
        with st.spinner("🤖 AI is analyzing..."):
            try:
                # EXACT COLUMN FORMAT FROM YOUR MODEL
                input_data = {
                    'Age': age, 'Gender': gender, 'Tenure': tenure,
                    'Usage Frequency': usage_freq, 'Support Calls': support_calls,
                    'Payment Delay': payment_delay, 'Subscription Type': subscription,
                    'Contract Length': contract, 'Total Spend': total_spend,
                    'Last Interaction': last_interaction, 'Support_Intensity': support_intensity,
                    'Avg_Spend_per_Month': avg_spend, 'Customer_Value': customer_value
                }
                
                processed_df = preprocess_input(input_data)
                probabilities = model.predict_proba(processed_df)[0]
                churn_prob = round(probabilities[1] * 100, 2)
                
                st.markdown('<div class="results-card">', unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f'<div class="metric-box"><div class="metric-label">Churn Probability</div><div class="metric-value">{churn_prob}%</div></div>', unsafe_allow_html=True)
                with col_m2:
                    risk = "HIGH" if churn_prob >= 70 else "MEDIUM" if churn_prob >= 40 else "LOW"
                    st.markdown(f'<div class="metric-box"><div class="metric-label">Risk Level</div><div class="metric-value">{risk}</div></div>', unsafe_allow_html=True)
                with col_m3:
                    confidence = max(churn_prob, 100-churn_prob)
                    st.markdown(f'<div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value">{confidence:.1f}%</div></div>', unsafe_allow_html=True)
                
                st.plotly_chart(create_gauge_chart(churn_prob), use_container_width=True)
                
                # --- SHINY GOLD HEADER ---
                st.markdown('<h3 class="gold-header">💡 Recommended Actions</h3>', unsafe_allow_html=True)
                
                for icon, text in generate_recommendations(churn_prob, input_data):
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <strong>{icon}</strong> {text}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
