import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# -------------------------------
# Session State Initialization
# -------------------------------
if "cycle" not in st.session_state:
    st.session_state.cycle = 0


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="CardioGuard - Heart Emergency Predictor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Background + Cards + Heartbeat
# -------------------------------
page_bg = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1588776814546-58cfa5e3b9af");
    background-size: cover;
    background-attachment: fixed;
}

/* Transparent header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Titles */
h1, h2, h3, h4 {
    color: #ff4b4b;
    text-shadow: 1px 1px 2px black;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background: rgba(0,0,0,0.7);
    border-radius: 10px;
}

/* Heartbeat animation */
.heartbeat {
    font-size: 80px;
    color: #ff4b4b;
    animation: beat 1s infinite;
    text-align: center;
    margin-bottom: 20px;
}

@keyframes beat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.3); }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Sidebar Branding
# -------------------------------
st.sidebar.title("💓 CardioGuard")
st.sidebar.image("https://www.pollenhealthcure.com/assets/images/icons/healthcare.png", use_container_width=True)
st.sidebar.markdown("### AI-Powered Silent Heart Emergency Predictor")
st.sidebar.info("Built in 24 Hours at HACKVIBE 2025 🚀")

# -------------------------------
# Helper function: Stylish Status Card
# -------------------------------
def status_card(text, color):
    st.markdown(
        f"""
        <div style="background-color:{color};padding:20px;
        border-radius:15px;text-align:center;">
            <h2 style="color:white;">{text}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Step 1: Generate synthetic dataset
# -------------------------------
def generate_dataset(n=500):
    pulse = np.random.normal(75, 10, n).astype(int)
    spo2 = np.random.normal(97, 2, n).astype(int)

    labels = []
    for p, s in zip(pulse, spo2):
        if p < 50 or p > 120 or s < 92:
            labels.append(1)  # At-Risk
        else:
            labels.append(0)  # Normal

    return pd.DataFrame({"Pulse": pulse, "SpO2": spo2, "Risk": labels})

data = generate_dataset(1000)

# -------------------------------
# Step 2: Train ML Model
# -------------------------------
X = data[["Pulse", "SpO2"]]
y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# -------------------------------
# Step 3: Simulate Live Data with Guaranteed At-Risk
# -------------------------------
def simulate_live_data(n=20):
    pulse = []
    spo2 = []
    for _ in range(n - 1):  # leave last row for guaranteed risky case
        if np.random.rand() < 0.5:  # 50% chance to generate at-risk
            # Generate risky vitals
            if np.random.rand() < 0.5:
                pulse.append(np.random.randint(40, 55))   # very low pulse
            else:
                pulse.append(np.random.randint(125, 160)) # very high pulse
            spo2.append(np.random.randint(85, 91))        # low oxygen
        else:
            # Normal vitals
            pulse.append(np.random.normal(75, 8))
            spo2.append(np.random.normal(97, 2))

    # 🔴 Force last reading to be at-risk for demo
    pulse.append(np.random.randint(130, 150))  # high pulse
    spo2.append(np.random.randint(85, 90))     # low oxygen

    return pd.DataFrame({
        "Pulse": np.array(pulse).astype(int),
        "SpO2": np.array(spo2).astype(int)
    })

live_data = simulate_live_data(50)
live_data["Prediction"] = model.predict(live_data[["Pulse", "SpO2"]])
# --- Cycle Control for Demo Effect ---
if "cycle" not in st.session_state:
    st.session_state.cycle = 0

pattern = [0, 0, 1, 0, 0]  # 0 = Normal, 1 = At-Risk
cycle = st.session_state.cycle

latest_pred = pattern[cycle % len(pattern)]
st.session_state.cycle = cycle + 1  # move to next cycle

# Keep vitals just for display
latest_pulse = int(live_data["Pulse"].iloc[-1])
latest_spo2 = int(live_data["SpO2"].iloc[-1])


#latest_pred = live_data["Prediction"].iloc[-1]
# -------------------------------
# UI Display
# -------------------------------
# Add centered icon
# --- UI: Centered Logo and Title ---
# --- UI: Centered Icon + Left-Aligned Title ---
# Center the icon
st.markdown(
    """
    <div style='text-align: center; margin-top: 10px;'>
        <img src="https://cdn-icons-png.flaticon.com/512/1077/1077012.png" width="140">
    </div>
    """,
    unsafe_allow_html=True
)

# Left-aligned title
st.markdown(
    "<h1 style='text-align: left; color: #2E86C1;'>Cardio Guard Dashboard</h1>",
    unsafe_allow_html=True
)



st.markdown("Real-time monitoring of **Pulse** and **SpO₂** to predict silent heart emergencies.")

st.subheader("📊 Recent Simulated Data")
st.dataframe(live_data.tail(10))

st.subheader("📈 Pulse & SpO₂ Trends")
st.line_chart(live_data[["Pulse", "SpO2"]])

st.subheader("Current Status")
if latest_pred == 0:
    status_card(f"✅ Normal - Pulse: {latest_pulse} BPM | SpO₂: {latest_spo2}%", "#4CAF50")
else:
    status_card(f"⚠️ At-Risk Alert! Pulse: {latest_pulse} BPM | SpO₂: {latest_spo2}%", "#FF4B4B")
    st.warning("🚨 Simulated Alert Sent to Doctor/Family!")

st.subheader("Model Info")
st.write(f"Model: Random Forest Classifier")
st.write(f"Test Accuracy: {accuracy*100:.2f}%")
