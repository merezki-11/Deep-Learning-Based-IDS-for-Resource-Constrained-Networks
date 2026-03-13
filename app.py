import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Group 4 IDS Dashboard", layout="wide")

# --- HEADER ---
st.title("🛡️ Deep Learning-Based IDS for Resource-Constrained Networks")
st.markdown("### Group 4 | Department of Cyber Security | NAU Awka")
st.divider()

# --- SIDEBAR CONTROL ---
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio("Active Model:", ("Full DNN", "Lightweight DNN"))

# --- DATA MAPPING (Matches your main.py logic) ---
results = {
    "Full DNN": {"acc": 80.90, "time": 185.78, "params": 13057, "latency": "High"},
    "Lightweight DNN": {"acc": 80.05, "time": 44.48, "params": 3457, "latency": "Low"}
}

current = results[model_type]

# --- SECTION 1: KEY PERFORMANCE INDICATORS ---
col1, col2, col3 = st.columns(3)
col1.metric("Detection Accuracy", f"{current['acc']}%")
col2.metric("Training Time", f"{current['time']}s")
col3.metric("Network Parameters", f"{current['params']:,}")

st.divider()

# --- SECTION 2: RESEARCH SIGNIFICANCE ---
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📊 Performance Analysis")
    # Comparison chart data
    chart_data = pd.DataFrame({
        'Model': ['Full DNN', 'Lightweight DNN'],
        'Accuracy': [80.90, 80.05],
        'Complexity (Params)': [13057, 3457]
    }).set_index('Model')
    st.bar_chart(chart_data['Accuracy'])
    st.caption("Comparison of detection accuracy across model variants.")

with right_col:
    st.subheader("💡 Resource Suitability")
    if model_type == "Lightweight DNN":
        st.success("✅ **RECOMMENDED:** This model is optimized for IoT/Embedded systems. It uses ~74% fewer parameters with only a 0.85% loss in accuracy.")
    else:
        st.warning("⚠️ **RESOURCE INTENSIVE:** This model provides the highest accuracy but requires significantly more memory and power, making it less ideal for sensor nodes.")

# --- SECTION 3: SIMULATED TRAFFIC ---
st.divider()
st.subheader("🔴 Simulated Real-Time Network Flow")
traffic = pd.DataFrame(np.random.randn(20, 2), columns=['Incoming Packets', 'Threat Score'])
st.line_chart(traffic)

st.info("**Significance:** This dashboard demonstrates that lightweight models can be deployed in constrained environments without critical security loss.")