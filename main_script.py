import streamlit as st
import pymupdf as fitz
import openai
from openai import OpenAI
import os
import tiktoken
import json
from googletrans import Translator

# Set your OpenAI API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
translator = Translator()

# -------- PDF & Text Utilities --------

def extract_text_from_pdf(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def split_text_into_chunks(text, max_tokens=2500):
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        tokens = len(enc.encode(" ".join(current_chunk)))
        if tokens >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------- GPT-Based MCQ Generation --------

def generate_mcqs(text, total_questions=5):
    prompt = f"""
You are a helpful assistant who generates clinically relevant multiple-choice questions (MCQs) strictly based on the provided text.
Make the questions clinically relevant to target an audience of medical students and residents, Royal College of Physicians and Surgeons of Canada style.
Do NOT write specific questions on case details, such as asking about a patient's blood pressure.
If the text refers to case numbers, do not add that information in the question stems.

Generate exactly {total_questions} MCQs in this JSON format:
[
  {{
    "question": "What is ...?",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A"
  }},
  ...
]

⚠️ Return ONLY valid JSON. No explanation or markdown.

TEXT:
\"\"\"
{text}
\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"⚠️ GPT MCQ generation failed: {e}")
        return []

# -------- Translation (GPT + Google Fallback) --------

def translate_mcqs(mcqs, language):
    if language == "English":
        return mcqs

    prompt = f"""
Translate the following multiple-choice questions into {language}, preserving the JSON structure:

{json.dumps(mcqs, indent=2)}

Return only the translated JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        st.warning(f"⚠️ GPT translation failed: {e}")
        st.info("🔁 Falling back to Google Translate...")

        try:
            translated_mcqs = []
            for item in mcqs:
                translated_item = {
                    "question": translator.translate(item["question"], dest=language).text,
                    "options": {
                        k: translator.translate(v, dest=language).text for k, v in item["options"].items()
                    },
                    "answer": item["answer"]  # keep original answer key (A/B/C/D)
                }
                translated_mcqs.append(translated_item)
            return translated_mcqs
        except Exception as ge:
            st.error(f"❌ Google Translate failed: {ge}")
            return mcqs

# -------- Quiz Scoring --------

def score_quiz(user_answers, original_mcqs):
    score = 0
    results = []
    for idx, user_ans in enumerate(user_answers):
        correct = original_mcqs[idx]["answer"]
        is_correct = user_ans == correct
        if is_correct:
            score += 1
        results.append({
            "question": original_mcqs[idx]["question"],
            "selected": user_ans,
            "correct": correct,
            "options": original_mcqs[idx]["options"],
            "is_correct": is_correct
        })
    return score, results

# -------- Streamlit App --------

st.set_page_config(page_title="PDF to MCQ Quiz", layout="centered")
st.title("📄 PDF to MCQ Quiz App")

# Language selector
st.markdown("### 🌐 Select Quiz Language")
language_map = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Ukrainian": "uk",
    "Russian": "ru",
    "German": "de",
    "Polish": "pl",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Hindi": "hi"
}
language_options = list(language_map.keys())
target_language_name = st.selectbox("Translate quiz to:", language_options, index=0)
target_language_code = language_map[target_language_name]

# File upload
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully.")
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.session_state["extracted_text"] = extracted_text

    with st.expander("🔍 Preview Extracted Text"):
        st.text_area("Extracted Text", extracted_text[:1000] + "...", height=300)

    total_questions = st.slider("🔢 Total number of MCQs to generate", 1, 20, 5)

    if st.button("🧠 Generate Quiz"):
        chunks = split_text_into_chunks(extracted_text)
        first_chunk = chunks[0] if chunks else extracted_text

        with st.spinner("Generating questions..."):
            mcqs = generate_mcqs(first_chunk, total_questions)
            st.session_state["original_mcqs"] = mcqs

        if mcqs:
            with st.spinner(f"Translating to {target_language_name}..."):
                translated_mcqs = translate_mcqs(mcqs, target_language_code)
                st.session_state["translated_mcqs"] = translated_mcqs
        else:
            st.error("❌ No MCQs were generated.")

# Quiz form
if st.session_state.get("translated_mcqs"):
    mcqs = st.session_state["translated_mcqs"]
    original_mcqs = st.session_state["original_mcqs"]
    user_answers = []

    with st.form("quiz_form"):
        st.header("📝 Take the Quiz")

        for idx, mcq in enumerate(mcqs):
            st.subheader(f"Q{idx + 1}: {mcq['question']}")
            options = mcq["options"]
            selected_text = st.radio(
                "Choose an answer:",
                list(options.values()),
                key=f"q{idx}"
            )
            selected_letter = next(k for k, v in options.items() if v == selected_text)
            user_answers.append(selected_letter)
            st.markdown("---")

        submitted = st.form_submit_button("✅ Submit Quiz")

    if submitted:
        score, results = score_quiz(user_answers, original_mcqs)
        st.success(f"🎯 You scored {score} out of {len(results)}")

        with st.expander("📊 View Detailed Feedback"):
            for i, r in enumerate(results):
                st.markdown(f"**Q{i+1}: {r['question']}**")
                for letter, text in r['options'].items():
                    if letter == r['correct']:
                        st.markdown(f"- ✅ **{letter}. {text}**")
                    elif letter == r['selected']:
                        st.markdown(f"- ❌ {letter}. {text}")
                    else:
                        st.markdown(f"- {letter}. {text}")
                st.markdown("---")
