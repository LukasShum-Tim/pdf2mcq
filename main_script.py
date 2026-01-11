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
import time
import re
import random
import io

# Set your OpenAI API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
translator = Translator()

# Reusing the quiz
if "quiz_version" not in st.session_state:
    st.session_state["quiz_version"] = 0

if "show_generate_new" not in st.session_state:
    st.session_state["show_generate_new"] = False

if "topics_initialized" not in st.session_state:
    st.session_state["topics_initialized"] = False

# -------- PDF & Text Utilities --------

def select_topics_for_quiz(n):
    topic_status = st.session_state["topic_status"]

    max_count = max(v["count"] for v in topic_status.values())
    weights = {
        t: (max_count + 1 - topic_status[t]["count"])
        for t in topic_status
    }

    return random.choices(
        list(weights.keys()),
        weights=list(weights.values()),
        k=n
    )
def extract_text_from_pdf(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_topics(text):
    prompt = f"""
You are analyzing a scientific or educational document.

Extract all major high-level topics or themes covered in this text.
Topics should be concise (1–4 words), non-overlapping, and clinically meaningful.

Return ONLY valid JSON in this format:
["Topic 1", "Topic 2", "Topic 3", ...]

TEXT:
\"\"\"{text[:12000]}\"\"\" 
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return json.loads(response.choices[0].message.content)

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

def shuffle_mcqs_pairwise(mcqs, translated_mcqs):
    synced_eng, synced_trans = [], []

    for eng_mcq, trans_mcq in zip(mcqs, translated_mcqs):
        options = eng_mcq["options"]
        correct_letter = eng_mcq["answer"]
        correct_text = options[correct_letter]

        keys = list(options.keys())
        random.shuffle(keys)

        new_eng_opts = {}
        new_trans_opts = {}
        for new_letter, old_letter in zip(string.ascii_uppercase, keys):
            new_eng_opts[new_letter] = eng_mcq["options"][old_letter]
            new_trans_opts[new_letter] = trans_mcq["options"][old_letter]

        new_correct = next(k for k, v in new_eng_opts.items() if v == correct_text)

        eng_mcq["options"] = new_eng_opts
        trans_mcq["options"] = new_trans_opts
        eng_mcq["answer"] = trans_mcq["answer"] = new_correct

        synced_eng.append(eng_mcq)
        synced_trans.append(trans_mcq)

    return synced_eng, synced_trans

# -------- GPT-Based MCQ Generation --------

def update_progress(progress, status, value, message):
    progress.progress(value)
    status.markdown(f"⏳ **{ui(message)}**")

def generate_mcqs(text, topics, seed_token=None):
    seed_token = seed_token or str(time.time())
    seed_text = f"\n\nSeed token: {seed_token}"

    topic_block = "\n".join([f"- {t}" for t in topics])
    total_questions = len(topics)
    
    prompt = f"""
You are a helpful assistant who generates clinically relevant multiple-choice questions (MCQs) strictly based on the provided text.
Make the questions clinically relevant to target an audience of residents, Royal College of Physicians and Surgeons of Canada style.
Focus on clinical relevance, and if surgical content exists, include surgical presentation, approach, and management.
Do NOT write specific questions on case details, such as asking about a patient's blood pressure.
If the text refers to case numbers, do not add that information in the question stems.

Rules:
- Generate EXACTLY one MCQ per topic listed
- Each question must focus ONLY on its assigned topic
- Do NOT repeat concepts across questions
- Do NOT invent content not present in the text

You MUST generate exactly {total_questions} MCQs.

TOPICS:
{topic_block}

Return ONLY valid JSON in the following format:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "answer": "A",
      "topic": "Exact topic name from list"
    }}
  ]
}}

⚠️ Return ONLY valid JSON. Do not include any text outside the JSON. Do not write anything else. No explanations, no extra text.

TEXT:
\"\"\"{text}\"\"\"
"""
    try:
        response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
        raw = response.choices[0].message.content
        
        # Safe JSON parsing
        match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Failed to extract JSON from GPT output:\n{raw}")

        return json.loads(match.group())["mcqs"]
        
    except Exception as e:
        st.warning(f"⚠️ GPT MCQ generation failed: {e}")
        return []

# -------- Translation (GPT + Google Fallback) --------

def translate_text_gpt(text, language_code):
    """Translate plain text using GPT, output as simple text."""
    try:
        # Include both name and code to ensure clarity for the model
        prompt = f"Translate the following text into the language corresponding to code '{language_code}'. Do not include explanations, only the translation:\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from GPT translation")
        return content.strip()
    except Exception as e:
        st.warning(f"⚠️ GPT translation failed for text: {e}")
        return None


async def translate_with_google(mcqs, language_code):
    """Use Google Translate to translate MCQs asynchronously."""
    translator = Translator()
    translated_mcqs = []
    for item in mcqs:
        translated_item = {
            "question": (await translator.translate(item["question"], dest=language_code)).text,
            "options": {
                k: (await translator.translate(v, dest=language_code)).text for k, v in item["options"].items()
            },
            "answer": item["answer"]
        }
        translated_mcqs.append(translated_item)
    return translated_mcqs


def translate_mcqs(mcqs, language_code):
    """Translate all MCQs using GPT, with Google fallback."""
    if language_code == "en":
        return mcqs

    translated_mcqs = []

    try:
        for mcq in mcqs:
            question_translated = translate_text_gpt(mcq["question"], language_code)
            if not question_translated:
                raise ValueError("Empty GPT translation")

            translated_options = {}
            for k, v in mcq["options"].items():
                option_translated = translate_text_gpt(v, language_code)
                if not option_translated:
                    raise ValueError("Empty GPT translation for option")
                translated_options[k] = option_translated

            translated_mcqs.append({
                "question": question_translated,
                "options": translated_options,
                "answer": mcq["answer"]
            })

        return translated_mcqs

    except Exception as e:
        st.warning(f"⚠️ GPT translation failed: {e}")
        st.info("🔁 Falling back to Google Translate...")
        try:
            translated_mcqs = asyncio.run(translate_with_google(mcqs, language_code))
            return translated_mcqs
        except Exception as ge:
            st.error(f"❌ Google Translate failed: {ge}")
            return mcqs

@st.cache_data(show_spinner=False)
def translate_ui_text(text, language_code):
    """Translate UI strings with caching."""
    if language_code == "en":
        return text
    translated = translate_text_gpt(text, language_code)
    return translated if translated else text

def ui(text):
    """
    Returns bilingual UI text:
    English
    Translated (if applicable)
    """
    if st.session_state.get("target_language_code", "en") == "en":
        return text

    translated = translate_ui_text(text, st.session_state["target_language_code"])

    if translated and translated != text:
        return f"{text} — {translated}"

    return text

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
    'English': 'en',
    'Afrikaans': 'af',
    'Albanian': 'sq',
    'Amharic': 'am',
    'Arabic': 'ar',
    'Armenian': 'hy',
    'Azerbaijani': 'az',
    'Basque': 'eu',
    'Belarusian': 'be',
    'Bengali': 'bn',
    'Bosnian': 'bs',
    'Bulgarian': 'bg',
    'Catalan': 'ca',
    'Cebuano': 'ceb',
    'Chichewa': 'ny',
    'Chinese (Simplified)': 'zh-cn',
    'Chinese (Traditional)': 'zh-tw',
    'Corsican': 'co',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'Esperanto': 'eo',
    'Estonian': 'et',
    'Filipino': 'tl',
    'Finnish': 'fi',
    'French': 'fr',
    'Frisian': 'fy',
    'Galician': 'gl',
    'Georgian': 'ka',
    'German': 'de',
    'Greek': 'el',
    'Gujarati': 'gu',
    'Haitian Creole': 'ht',
    'Hausa': 'ha',
    'Hawaiian': 'haw',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Hmong': 'hmn',
    'Hungarian': 'hu',
    'Icelandic': 'is',
    'Igbo': 'ig',
    'Indonesian': 'id',
    'Irish': 'ga',
    'Italian': 'it',
    'Japanese': 'ja',
    'Javanese': 'jw',
    'Kannada': 'kn',
    'Kazakh': 'kk',
    'Khmer': 'km',
    'Korean': 'ko',
    'Kurdish (Kurmanji)': 'ku',
    'Kyrgyz': 'ky',
    'Lao': 'lo',
    'Latin': 'la',
    'Latvian': 'lv',
    'Lithuanian': 'lt',
    'Luxembourgish': 'lb',
    'Macedonian': 'mk',
    'Malagasy': 'mg',
    'Malay': 'ms',
    'Malayalam': 'ml',
    'Maltese': 'mt',
    'Maori': 'mi',
    'Marathi': 'mr',
    'Mongolian': 'mn',
    'Myanmar (Burmese)': 'my',
    'Nepali': 'ne',
    'Norwegian': 'no',
    'Odia': 'or',
    'Pashto': 'ps',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Punjabi': 'pa',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Samoan': 'sm',
    'Scots Gaelic': 'gd',
    'Serbian': 'sr',
    'Sesotho': 'st',
    'Shona': 'sn',
    'Sindhi': 'sd',
    'Sinhala': 'si',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Somali': 'so',
    'Spanish': 'es',
    'Sundanese': 'su',
    'Swahili': 'sw',
    'Swedish': 'sv',
    'Tajik': 'tg',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Thai': 'th',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Uyghur': 'ug',
    'Uzbek': 'uz',
    'Vietnamese': 'vi',
    'Welsh': 'cy',
    'Xhosa': 'xh',
    'Yiddish': 'yi',
    'Yoruba': 'yo',
    'Zulu': 'zu',
}
language_options = list(language_map.keys())
target_language_name = st.selectbox(
    ui("Translate quiz to:"),
    language_options,
    index=0,
    key="language_selector"
)
target_language_code = language_map[target_language_name]
st.session_state["target_language_code"] = target_language_code

#Building the quiz
def build_quiz():
    progress = st.progress(0)
    status = st.empty()
    st.session_state["quiz_saved"] = False
    
    update_progress(progress, status, 5, "Preparing quiz...")

    # Clear old answers
    for k in list(st.session_state.keys()):
        if k.startswith("q_"):
            del st.session_state[k]
    
    update_progress(progress, status, 15, "Generating questions with AI...")
    
    selected_topics = select_topics_for_quiz(st.session_state["total_questions"])
    
    mcqs = generate_mcqs(
        st.session_state["extracted_text"],
        topics=selected_topics,
        seed_token=str(time.time())
    )

    for mcq in mcqs:
        st.session_state["topic_status"][mcq["topic"]]["count"] += 1
        st.session_state["topic_status"][mcq["topic"]]["questions"].append(mcq)
            
    update_progress(progress, status, 35, "Tracking covered topics...")

    if st.session_state["target_language_code"] != "en":
        update_progress(progress, status, 55, "Translating questions...")
            
    translated = translate_mcqs(mcqs, st.session_state["target_language_code"])

    update_progress(progress, status, 75, "Shuffling answer choices...")
    
    mcqs, translated = shuffle_mcqs_pairwise(mcqs, translated)

    update_progress(progress, status, 90, "Finalizing quiz...")
    
    st.session_state["original_mcqs"] = mcqs
    st.session_state["translated_mcqs"] = translated
    st.session_state["quiz_version"] += 1
    st.session_state["show_results"] = False
    st.session_state["show_generate_new"] = False

    update_progress(progress, status, 100, "Quiz ready!")

    time.sleep(0.3)  # UX polish
    progress.empty()
    status.empty()

# File upload

uploaded_file = st.file_uploader(
    ui("📤 Upload your PDF file. If using a mobile device, please make sure the PDF file is stored on your local drive, and not imported from a cloud drive to prevent upload errors."),
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_file:
    st.session_state["pdf_bytes"] = uploaded_file.getvalue()
    st.session_state["pdf_changed"] = True
    st.session_state["topics_initialized"] = False

if "pdf_bytes" in st.session_state:
    extracted_text = extract_text_from_pdf(
        io.BytesIO(st.session_state["pdf_bytes"])
    )
    st.session_state["extracted_text"] = extracted_text

    st.success(ui("✅ PDF uploaded successfully."))

    with st.expander(ui("🔍 Preview Extracted Text")):
        st.text_area("Extracted Text", extracted_text[:1000] + "...", height=300)

    st.slider(
        ui("🔢 Total number of MCQs to generate"),
        1, 20, 5,
        key="total_questions"
    )

    if not st.session_state["topics_initialized"]:
        st.session_state["topics"] = extract_topics(extracted_text)
        st.session_state["topic_status"] = {
            t: {"count": 0, "questions": []}
            for t in st.session_state["topics"]
        }
        st.session_state["topics_initialized"] = True

    if st.button(ui("🧠 Generate Quiz")):
        build_quiz()

if st.session_state.get("translated_mcqs"):
    translated_mcqs = st.session_state["translated_mcqs"]
    original_mcqs = st.session_state["original_mcqs"]
    user_answers = []

    bilingual_mode = st.session_state["target_language_code"] != "en"

    with st.form(f"quiz_form_{st.session_state['quiz_version']}"):
        st.header(ui("📝 Take the Quiz"))

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
                bilingual_opts = [
                    f"{options[k]}  \n*EN: {english_opts[k]}*"
                    for k in ordered_keys
                ]
                selected_text = st.radio(
                    "Choose an answer:",
                    bilingual_opts,
                    key=f"q_{st.session_state['quiz_version']}_{idx}"
                )
                selected_letter = ordered_keys[bilingual_opts.index(selected_text)]
            else:
                ordered_options = [options[k] for k in ordered_keys]
                selected_text = st.radio(
                    "Choose an answer:",
                    ordered_options,
                    key=f"q_{st.session_state['quiz_version']}_{idx}"
                )
                selected_letter = next(
                    k for k, v in options.items() if v == selected_text
                )

            user_answers.append(selected_letter)
            st.markdown("---")

        submitted = st.form_submit_button(ui("✅ Submit Quiz"))


    if submitted:
        st.session_state["show_results"] = True
        st.session_state["show_generate_new"] = True

    if st.session_state.get("show_results"):
        score, results = score_quiz(
            user_answers,
            translated_mcqs,
            original_mcqs
        )
        st.success(ui(f"🎯 You scored {score} out of {len(results)}"))

        #Question history
        if "quiz_history" not in st.session_state:
            st.session_state["quiz_history"] = []
        
        # Prevent duplicate save on reruns
        if not st.session_state.get("quiz_saved", False):
            st.session_state["quiz_history"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "language_name": st.session_state["target_language_code"],
                "language_code": target_language_code,
                "score": score,
                "total": len(results),
                "questions": [
                    {
                        "english": {
                            "question": original_mcqs[i]["question"],
                            "options": original_mcqs[i]["options"],
                        },
                        "translated": {
                            "question": translated_mcqs[i]["question"],
                            "options": translated_mcqs[i]["options"],
                        },
                        "correct": original_mcqs[i]["answer"],
                        "selected": user_answers[i]
                    }
                    for i in range(len(original_mcqs))
                ]
            })
            st.session_state["quiz_saved"] = True
        
        # -------- Feedback section --------
        with st.expander(ui("📊 View Detailed Feedback")):
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


        #Quiz history drop-down menu
        st.markdown(ui("📂 Quiz History & Topic Coverage"))
    
        view_mode = st.selectbox(
            ui("Select one of the options below:"),
            [
                ui("Major Topics"),
                ui("Previous Questions"),
            ]
        )
    
        if view_mode.startswith("**Major Topics**") or view_mode.startswith("Major Topics"):
            for topic, data in st.session_state["topic_status"].items():
                if data["count"] == 0:
                    status_text = ui("⏳ Not yet asked")
                else:
                    status_text = ui(f"Asked {data['count']} time(s)")
                
                st.markdown(f"**{topic}** — {status_text}")
    
        elif view_mode.startswith("**Previous Questions**") or view_mode.startswith("Previous Questions"):
            quiz_idx = st.selectbox(
                ui("Select quiz attempt:"),
                list(range(len(st.session_state["quiz_history"]))),
                format_func=lambda i: f"Attempt {i + 1} — {st.session_state['quiz_history'][i]['timestamp']}"
            )
        
            quiz = st.session_state["quiz_history"][quiz_idx]
            
            bilingual_mode = target_language_code != "en"
            
            st.markdown(
                f"{ui('Score')}: {quiz['score']}/{quiz['total']} "
                f"({quiz['language_name']})"
            )
            
            for i, q in enumerate(quiz["questions"]):
                if bilingual_mode:
                    st.markdown(f"### Q{i + 1}: {q['translated']['question']}")
                    st.caption(f"**English:** {q['english']['question']}")
                    options = q["translated"]["options"]
                    english_opts = q["english"]["options"]
                else:
                    st.markdown(f"### Q{i + 1}: {q['english']['question']}")
                    options = q["english"]["options"]
                    english_opts = None
            
                for letter, text in options.items():
                    if letter == q["correct"]:
                        st.markdown(f"- ✅ **{letter}. {text}**")
                        if bilingual_mode:
                            st.caption(f"  **EN:** {english_opts[letter]}")
                    elif letter == q["selected"]:
                        st.markdown(f"- ❌ {letter}. {text}")
                        if bilingual_mode:
                            st.caption(f"  **EN:** {english_opts[letter]}")
                    else:
                        st.markdown(f"- {letter}. {text}")
                        if bilingual_mode:
                            st.caption(f"  **EN:** {english_opts[letter]}")
            
                st.markdown("---")

    
#Generate new questions
if st.session_state.get("show_generate_new"):
    if st.button(ui("🔄 Generate New Questions")):        
        build_quiz()
        st.rerun()

#Feedback Form
    url_instructors = "https://forms.gle/GdMqpvikomBRTcvJ6"
    url_students = "https://forms.gle/CWKRqptQhpdLKaj8A"

    st.markdown(ui("📝 Help Us Improve"))
    # Show translation only if non-English
    if target_language_code != "en":
        translated_feedback = translate_text_gpt(
            "Help us improve",
            target_language_code
        )
        if translated_feedback:
            st.write(f"**{translated_feedback}**")
    
    # Always show English
    feedback_text_en = (
    "Thank you for trying this multilingual short answer question generator! "
    "Please click on the following links to provide feedback to help improve this tool:"
    )
    
    feedback_instructors_en = "Feedback form for instructors:"
    feedback_students_en = "Feedback form for students:"
    st.write(feedback_text_en)

    # Show translation only if non-English
    if target_language_code != "en":
        translated_feedback = translate_text_gpt(
            feedback_text_en,
            target_language_code
        )
        if translated_feedback:
            st.write(translated_feedback)

    st.markdown("---")

    # Instructor feedback
    st.markdown(f"**{feedback_instructors_en}**")
    if target_language_code != "en":
        st.caption(
            translate_text_gpt(
                feedback_instructors_en,
                target_language_code
            )
        )
    st.markdown(url_instructors)

    # Student feedback
    st.markdown(f"**{feedback_students_en}**")
    if target_language_code != "en":
        st.caption(
            translate_text_gpt(
                feedback_students_en,
                target_language_code
            )
        )
    st.markdown(url_students)
