import streamlit as st
import pymupdf as fitz
import openai
from openai import OpenAI
import os
import tiktoken
import json
from googletrans import Translator
import random
import asyncio
import string

# Set your OpenAI API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
translator = Translator()

# -------- PDF & Text Utilities --------

def extract_text_from_pdf(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# -------- GPT-Based MCQ Generation --------

def shuffle_options(mcq):
    """Shuffle the options' *values*, but reassign them back to keys A–D in order."""
    options = mcq['options']
    correct_letter = mcq['answer']
    correct_text = options[correct_letter]

    # Shuffle only the option texts (not the keys)
    shuffled_texts = list(options.values())
    random.shuffle(shuffled_texts)

    # Reassign shuffled texts to new keys A–D
    new_letters = list(string.ascii_uppercase[:len(shuffled_texts)])
    new_options = {letter: text for letter, text in zip(new_letters, shuffled_texts)}

    # Find which new letter now holds the original correct text
    new_correct_letter = next(letter for letter, text in new_options.items() if text == correct_text)

    mcq['options'] = new_options
    mcq['answer'] = new_correct_letter
    return mcq

# -------- GPT-Based MCQ Generation --------

def generate_mcqs(text, total_questions=5):
    prompt = f"""
You are a helpful assistant who generates clinically relevant multiple-choice questions (MCQs) strictly based on the provided text.
Make the questions clinically relevant to target an audience of medical students and residents, Royal College of Physicians and Surgeons of Canada style.
Ensure the questions are **proportional across the manual**, covering all major topics.
Focus on clinical relevance, and if surgical content exists, include surgical presentation, approach, and management.
Do NOT write specific questions on case details, such as asking about a patient's blood pressure.
If the text refers to case numbers, do not add that information in the question stems.

Generate exactly {total_questions} MCQs in this JSON format:
[{{"question": "What is ...?", "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}}, "answer": "A"}}, ...]

⚠️ Return ONLY valid JSON. No explanation or markdown.

TEXT:
\"\"\"{text}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        mcqs = json.loads(response.choices[0].message.content)
        
        # Shuffle options for each MCQ
        for mcq in mcqs:
            mcq = shuffle_options(mcq)

        return mcqs
    except Exception as e:
        st.warning(f"⚠️ GPT MCQ generation failed: {e}")
        return []


# -------- Translation (GPT + Google Fallback) --------

async def translate_with_google(mcqs, language):
    """Use Google Translate to translate MCQs asynchronously."""
    translator = Translator()
    translated_mcqs = []
    for item in mcqs:
        translated_item = {
            "question": (await translator.translate(item["question"], dest=language)).text,
            "options": {
                k: (await translator.translate(v, dest=language)).text for k, v in item["options"].items()
            },
            "answer": item["answer"]  # keep original answer key (A/B/C/D)
        }
        translated_mcqs.append(translated_item)
    return translated_mcqs


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
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        translated_mcqs = json.loads(response.choices[0].message.content)

        # Check if GPT returned valid MCQs
        if not translated_mcqs:
            raise ValueError("GPT returned empty response")

        # After translation, shuffle options in the same way to keep the answer key intact
        for mcq in translated_mcqs:
            mcq = shuffle_options(mcq)

        return translated_mcqs

    except Exception as e:
        st.warning(f"⚠️ GPT translation failed: {e}")
        st.info("🔁 Falling back to Google Translate...")

        # Fallback to Google Translate asynchronously
        try:
            translated_mcqs = asyncio.run(translate_with_google(mcqs, language))
            
            # Shuffle options after translation
            for item in translated_mcqs:
                item = shuffle_options(item)

            return translated_mcqs

        except Exception as ge:
            st.error(f"❌ Google Translate failed: {ge}")
            return mcqs

# -------- Quiz Scoring --------

def score_quiz(user_answers, translated_mcqs, original_mcqs=None):
    """
    Scores the quiz based on the translated MCQs (the ones shown to the user).
    Optionally, includes English reference if original_mcqs is provided.
    """
    score = 0
    results = []

    for idx, user_ans in enumerate(user_answers):
        mcq = translated_mcqs[idx]
        correct = mcq["answer"]
        is_correct = user_ans == correct
        if is_correct:
            score += 1

        result = {
            "question": mcq["question"],
            "selected": user_ans,
            "correct": correct,
            "options": mcq["options"],
            "is_correct": is_correct
        }

        # Optional: include English reference for bilingual review
        if original_mcqs:
            result["english_question"] = original_mcqs[idx]["question"]
            result["english_options"] = original_mcqs[idx]["options"]

        results.append(result)

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
        full_text = extracted_text
    
        with st.spinner("Generating questions..."):
            mcqs = generate_mcqs(full_text, total_questions)
            st.session_state["original_mcqs"] = mcqs
    
        if mcqs:
            with st.spinner(f"Translating to {target_language_name}..."):
                translated_mcqs = translate_mcqs(mcqs, target_language_code)
                st.session_state["translated_mcqs"] = translated_mcqs
        else:
            st.error("❌ No MCQs were generated.")

# Quiz form
if st.session_state.get("translated_mcqs"):
    translated_mcqs = st.session_state["translated_mcqs"]
    original_mcqs = st.session_state["original_mcqs"]
    user_answers = []

    bilingual_mode = target_language_name != "English"

with st.form("quiz_form"):
    st.header("📝 Take the Quiz")

    bilingual_mode = target_language_name != "English"

    for idx, mcq in enumerate(translated_mcqs):
        if bilingual_mode:
            st.markdown(f"### Q{idx + 1}: {mcq['question']}")
            st.caption(f"**English:** {original_mcqs[idx]['question']}")
        else:
            st.subheader(f"Q{idx + 1}: {mcq['question']}")

        options = mcq["options"]
        ordered_keys = sorted(options.keys())

        if bilingual_mode:
            english_opts = original_mcqs[idx]["options"]
            bilingual_opts = []
            for k in ordered_keys:
                translated = options[k]
                english = english_opts[k]
                bilingual_opts.append(f"{translated}  \n*EN: {english}*")
            selected_text = st.radio(
                "Choose an answer:",
                bilingual_opts,
                key=f"q{idx}"
            )

            # Match the selected_text to its corresponding letter
            selected_letter = ordered_keys[bilingual_opts.index(selected_text)]
        else:
            ordered_options = [options[k] for k in ordered_keys]
            selected_text = st.radio(
                "Choose an answer:",
                ordered_options,
                key=f"q{idx}"
            )
            selected_letter = next(k for k, v in options.items() if v == selected_text)

        user_answers.append(selected_letter)
        st.markdown("---")

    submitted = st.form_submit_button("✅ Submit Quiz")

    if submitted:
        score, results = score_quiz(
            user_answers,
            translated_mcqs,
            original_mcqs
        )
        st.success(f"🎯 You scored {score} out of {len(results)}")

        # -------- Feedback section --------
        with st.expander("📊 View Detailed Feedback"):
            for i, r in enumerate(results):
                if bilingual_mode:
                    st.markdown(f"### Q{i+1}: {r['question']}")
                    st.caption(f"**English:** {r['english_question']}")
                else:
                    st.markdown(f"**Q{i+1}: {r['question']}**")

                for letter, text in sorted(r['options'].items()):
                    if letter == r['correct']:
                        st.markdown(f"- ✅ **{letter}. {text}**")
                        if bilingual_mode:
                            st.caption(f"  **EN:** {r['english_options'][letter]}")
                    elif letter == r['selected']:
                        st.markdown(f"- ❌ {letter}. {text}")
                        if bilingual_mode:
                            st.caption(f"  **EN:** {r['english_options'][letter]}")
                    else:
                        st.markdown(f"- {letter}. {text}")
                        if bilingual_mode:
                            st.caption(f"  **EN:** {r['english_options'][letter]}")
                st.markdown("---")
