import os

import pandas as pd
import streamlit as st
from joblib import load

from src.features import select_features

st.set_page_config(page_title="F1 Tire Degradation Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ F1 Tire Degradation Predictor")
st.write("Enter conditions and get a predicted tire degradation value.")

out_dir = st.sidebar.text_input("Model folder", value="outputs")
model_path = os.path.join(out_dir, "model.joblib")

if not os.path.exists(model_path):
    st.warning("Model not found. Train first: `python -m src.train --csv_path <path>`")
    st.stop()

pipeline = load(model_path)
event_categories = list(
    pipeline.named_steps["preprocess"].named_transformers_["event"].categories_[0]
)

tire_wear = st.slider("Tire wear", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
ambient_temp = st.number_input(
    "Ambient temperature (°C)", min_value=-10.0, max_value=60.0, value=22.0, step=0.5
)
event = (
    st.selectbox("Event", options=event_categories)
    if event_categories
    else st.text_input("Event")
)

df = pd.DataFrame([{
    "Tire_wear": tire_wear,
    "Humidity": humidity,
    "Ambient_Temperature": ambient_temp,
    "Event": event,
}])

X = select_features(df)
pred = float(pipeline.predict(X)[0])

st.metric("Predicted Tire Degradation", f"{pred:.4f}")
