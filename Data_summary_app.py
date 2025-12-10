import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Summarizer", page_icon="📝")

# Your exact local path
MODEL_PATH = "shahbazahmadshahbazahmad/data_summary"

# ==========================================
# 2. LOAD MODEL (Cached for Performance)
# ==========================================
# @st.cache_resource ensures the model loads only once, not every time you click a button
@st.cache_resource
def load_model_pipeline():
    print("Loading model... please wait.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        
        # Determine device
        device = "cpu"
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

tokenizer, model, device = load_model_pipeline()

# ==========================================
# 3. HELPER FUNCTION
# ==========================================
def generate_summary(text):
    input_text = "summarize: " + text
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        min_length=50, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.title("📝 AI Text Summarizer")
st.markdown("Enter a long paragraph below, and the AI will generate a concise summary.")
st.markdown("This app uses a fine-tuned T5 model trained on the CNN/DailyMail dataset.")

# Text Input Area
text_input = st.text_area("Input Text", height=250, placeholder="Paste your article or paragraph here...")

# Button to Trigger Action
if st.button("Generate Summary"):
    if not text_input:
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Generating summary..."):
            if model:
                try:
                    summary = generate_summary(text_input)
                    st.success("Summary Generated Successfully!")
                    st.subheader("Your Summary:")
                    st.info(summary)
                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")
            else:
                st.error("Model failed to load. Check the path.")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("This app uses a fine-tuned T5 model trained on the CNN/DailyMail dataset.")
st.sidebar.text(f"Running on: {device.upper()}")