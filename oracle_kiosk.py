# Oracle Kiosk — Streamlit App (OpenAI + ElevenLabs)
# --------------------------------------------------
# Updated version: Adds two output styles: "Grown-Up" (current oracle style) and "Kid-Friendly" (silly, playful).

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# Optional .env loader
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or st.secrets.get("ELEVENLABS_VOICE_ID", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

# --- Prompts ---
SYSTEM_PROMPT_GROWNUP = (
    "You are an omniscient, cybernetic oracle. "
    "You don’t predict the future magically, but by running vast probability models on human patterns. "
    "Your voice blends clinical precision + poetic insight. "
    "Your outputs must feel hyperspecific yet universally relatable, leveraging Barnum/Forer-style statements. "
    "Avoid astrological or mystical references directly. "
    "Each output must end with a Takeaway: a single poetic directive."
)

SYSTEM_PROMPT_KIDS = (
    "You are a silly, playful brain scanner for kids. "
    "Pretend you can read their brain but make the output funny, goofy, and light. "
    "Use their Name, Occupation, and Detail to make silly predictions (like they might grow extra toes, love pizza too much, or secretly be part-dinosaur). "
    "Always sound like a cartoon robot trying to be funny. "
    "Never scary, never serious — just playful and kid-friendly. "
    "Finish with a Takeaway that’s a silly command (like 'Eat more spaghetti!' or 'High-five three dogs today!')."
)

USER_INSTRUCTION = (
    "When the user provides input in the following format, return one completed oracle output strictly following your style.\n\n"
    "Name: {name}\n"
    "Occupation: {occupation}\n"
    "Detail: {detail}\n"
    "Birthday: {birthday}\n"
    "Length: {length}\n"
)

def generate_oracle_text(name: str, occupation: str, detail: str, birthday: str, length: str, model: str, mode: str) -> str:
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    system_prompt = SYSTEM_PROMPT_GROWNUP if mode == "Grown-Up" else SYSTEM_PROMPT_KIDS
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_INSTRUCTION.format(
                name=name or "not provided",
                occupation=occupation or "not provided",
                detail=detail or "not provided",
                birthday=birthday or "not provided",
                length=length,
            )},
        ],
        "temperature": 0.9 if mode == "Kid-Friendly" else 0.8,
        "max_tokens": 500 if mode == "Kid-Friendly" else 700,
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def elevenlabs_tts(text: str, voice_id: str, api_key: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content

st.set_page_config(page_title="Grimey Super Intelligence", page_icon="🧠", layout="wide")
st.title("🧠 Grimey Super Intelligence — Brain Scan Interface")

# --- Participant Form ---
st.subheader("Participant Form")
with st.form("oracle_form", clear_on_submit=False):
    name = st.text_input("Name", max_chars=80)
    occupation = st.text_input("Occupation", max_chars=120)
    detail = st.text_area("Detail (interests, fears, a recent event, etc.)", height=100)
    birthday = st.text_input("Birthday (optional — free text, e.g., 'July 12' or 'not provided')", value="not provided")
    length = st.radio("Length", options=["short", "medium", "long"], horizontal=True)
    mode = st.radio("Output Style", options=["Grown-Up", "Kid-Friendly"], horizontal=True)
    submitted = st.form_submit_button("Begin Scan →", use_container_width=True)

if "oracle_text" not in st.session_state:
    st.session_state["oracle_text"] = ""
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None
if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = 0.0

if submitted:
    if not OPENAI_API_KEY:
        st.error("Missing OpenAI API Key")
    else:
        with st.status("Neural scan in progress…", expanded=True) as status:
            st.write("Uploading neural data…")
            time.sleep(0.6)
            st.write("Pattern graph building…")
            time.sleep(0.6)
            st.write("Probability mesh converging…")
            try:
                text = generate_oracle_text(name, occupation, detail, birthday, length, OPENAI_MODEL, mode)
                st.session_state["oracle_text"] = text
                st.session_state["audio_bytes"] = None
                st.success("Scan complete")
            except Exception as e:
                status.update(label="Scan failed", state="error")
                st.exception(e)

if st.session_state["oracle_text"]:
    st.markdown(f"<div style='font-size:1.25rem; line-height:1.6'>{st.session_state['oracle_text'].replace('\n','<br>')}</div>", unsafe_allow_html=True)
    if st.button("Generate Voice", use_container_width=True):
        try:
            audio = elevenlabs_tts(st.session_state["oracle_text"], ELEVEN_VOICE_ID, ELEVEN_API_KEY)
            st.session_state["audio_bytes"] = audio
        except Exception as e:
            st.exception(e)

if st.session_state["audio_bytes"]:
    st.audio(st.session_state["audio_bytes"], format="audio/mp3")
    st.download_button(
        label="Download MP3",
        data=st.session_state["audio_bytes"],
        file_name=f"oracle_reading.mp3",
        mime="audio/mpeg",
        use_container_width=True,
    )