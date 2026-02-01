import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="PlantVillage FL Monitor", page_icon="🌱", layout="wide")

# Stylish Header
st.markdown("""
    <div style='background-color: #2e7d32; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <h1 style='color: white; margin: 0;'>🌱 PlantVillage MLOps Portal</h1>
        <p style='color: #e8f5e9; margin: 0;'>Federated Learning Model Monitoring Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

metrics_file = "outputs/logs/federated_metrics.csv"

if os.path.exists(metrics_file):
    df = pd.read_csv(metrics_file)
    
    # 1. Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    final_acc = df['accuracy'].iloc[-1]
    initial_acc = df['accuracy'].iloc[0]
    
    col1.metric("Final Accuracy", f"{final_acc*100:.2f}%", f"{(final_acc - initial_acc)*100:.2f}%")
    col2.metric("Federated Rounds", len(df))
    col3.metric("System Status", "Healthy", delta_color="normal")

    # 2. Convergence Plot
    st.subheader("Convergence Analysis")
    fig = px.line(df, x='round', y='accuracy', title='Global Model Accuracy (Interactive)',
                 markers=True, template="plotly_white", color_discrete_sequence=['#2e7d32'])
    fig.update_layout(xaxis_title="Communication Round", yaxis_title="Accuracy (0-1)")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Raw Data
    with st.expander("📂 View Detailed Round Metrics"):
        st.table(df)
else:
    st.warning("Awaiting metrics file... Run the Kaggle simulation to generate data.")
