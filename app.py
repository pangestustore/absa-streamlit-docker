import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from huggingface_hub import login
import os
import streamlit.components.v1 as components

# (Opsional) login ke Hugging Face jika pakai model privat
HF_TOKEN = os.getenv("HF_TOKEN", default=None)
if HF_TOKEN:
    login(token=HF_TOKEN)

# Path model dari Hugging Face
model_path = "pangestuu/indobert_sentiment_aspek"

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label
sentiment_labels = ['Negative', 'Positive']
aspect_labels = [
    "Akses Layanan Kesehatan Online",
    "Kemudahan Akses dan Kinerja Aplikasi",
    "Kendala Login dan Pembaruan",
    "Kendala Verifikasi dan OTP",
    "Kesulitan Penggunaan Aplikasi",
    "Manajemen Data dan Faskes"
]

def predict(text, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs).item()
        conf = torch.max(probs).item()
    return labels[pred_class], conf

# Streamlit UI
st.set_page_config(page_title="Analisis Ulasan Mobile JKN", layout="centered")
st.title("🩺 Analisis Ulasan Mobile JKN")
st.markdown("Masukkan ulasan dari pengguna aplikasi untuk memprediksi **aspek** dan **sentimen**.")

text = st.text_area("📝 Masukkan Ulasan Pengguna Aplikasi")

if st.button("🔍 Analisis"):
    if not text.strip():
        st.warning("Tolong masukkan teks ulasan terlebih dahulu.")
    else:
        aspect, conf_aspect = predict(text, aspect_labels)
        sentiment, conf_sent = predict(text, sentiment_labels)

        st.subheader("📊 Hasil Analisis:")
        st.markdown(f"- **Aspek**: `{aspect}` (Confidence: {conf_aspect:.2f})")
        st.markdown(f"- **Sentimen**: `{sentiment}` (Confidence: {conf_sent:.2f})")

st.title("📊 Visualisasi Looker Studio")
st.components.v1.iframe(
    "https://lookerstudio.google.com/embed/reporting/85cad12e-ebf2-473b-ac92-bb007c50d76b/page/FGdHF",
    height=600,
    width=800
)
