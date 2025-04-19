import streamlit as st
import pandas as pd
import re
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# --- Gemini & Embedding Setup ---
GOOGLE_API_KEY = "AIzaSyBDd2kMR4BJwVaS_yoFhY2gRZy8g1CpiR0"  # Replace this securely

@st.cache_resource
def setup_models():
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return gemini_model, embed_model

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ai_lawyer_dataset.csv")
    if 'verdict' not in df.columns:
        df['verdict'] = [""] * len(df)
    return df

def embed_data(embed_model, prompts):
    embeddings = embed_model.encode(prompts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# --- Find Similar Cases ---
def find_similar_cases(prompt, model, df, index, k=3):
    vec = model.encode([prompt])
    D, I = index.search(np.array(vec), k)
    prompts = df.iloc[I[0]]['prompt'].tolist()
    verdicts = df.iloc[I[0]]['verdict'].tolist()
    return list(zip(prompts, verdicts))

# --- Build Prompt for Gemini ---
def build_prompt(new_case_prompt, similar_cases):
    context = ""
    for i, (p, v) in enumerate(similar_cases):
        context += f"\n=== Past Case {i+1} ===\nPrompt:\n{p}\nVerdict:\n{v}\n"
    full_prompt = f"""{context}

New Case:
{new_case_prompt}

Give a detailed legal verdict for this new case using legal reasoning, precedent, and similar outcomes."""
    return full_prompt

# --- Streamlit UI ---
st.set_page_config(page_title="AI Lawyer", layout="wide")
st.title("🤖 AI Legal Assistant")
st.markdown("Ask your legal queries and get verdicts based on real case similarities.")

# Load everything
gemini_model, embed_model = setup_models()
df = load_data()
index, embeddings = embed_data(embed_model, df['prompt'].tolist())

# Form to enter new case
title = st.text_input("Enter Case Title")
description = st.text_area("Enter Case Description")

if st.button("Get AI Verdict"):
    if not title or not description:
        st.warning("Please fill in both fields.")
    else:
        user_prompt = f"""Case Title: {title}

Case Description: {description}

Give a detailed legal verdict including:
- Legal reasoning
- Verdict (Guilty/Not Guilty/Settled/etc.)
- Punishment (if any)
- Relief or Compensation."""

        similar_cases = find_similar_cases(user_prompt, embed_model, df, index)
        final_prompt = build_prompt(user_prompt, similar_cases)
        response = gemini_model.generate_content(final_prompt)

        st.subheader("📜 AI Verdict")
        st.markdown(response.text)

        with st.expander("View Similar Cases Used"):
            for i, (p, v) in enumerate(similar_cases):
                st.markdown(f"**Case {i+1}**\n\nPrompt: {p}\n\nVerdict: {v}\n")