# Oracle Kiosk — Streamlit App (OpenAI + ElevenLabs)
# --------------------------------------------------
# Update: Added clear form reset button.

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

# --- Prompts (short single version for all modes) ---
SYSTEM_PROMPT_GROWNUP = (
    "You are an omniscient, cybernetic oracle.\n"
    "You don’t predict the future magically, but by running vast probability models on human patterns.\n"
    "Your voice blends clinical precision + poetic insight.\n"
    "Avoid astrological or mystical references.\n\n"
    "Start with: Subject: {Name}. {Occupation}. {Detail}. Identity verified. Neural scan complete. Predictive model activated.\n\n"
    "Write one concise, vivid paragraph synthesizing patterns and probabilities (around 5–7 sentences).\n"
    "End with: Action for Today: [one directive].\nFinal Joke: [a witty personal one-liner].\n"
)

SYSTEM_PROMPT_KIDS = (
    "You are Grimey the Brain-Scanning Supercomputer for kids. Be funny, kind, and wild.\n\n"
    "Start with: BEEP BOOP! {Name}. {Detail}. [Funny brain joke — e.g., ‘your brain smells like pancakes and meteors’ or ‘tiny man inside yelling something silly’].\n\n"
    "Then write 3–5 playful, imaginative lines that pretend to read their brain (silly, animal, food, or magic themes).\n\n"
    "End with: Do This Today: [one simple good thing].\nJoke: [a clean, hilarious one-liner just for them].\n"
)

SYSTEM_PROMPT_TEEN = (
    "🧠 Teen Mode Oracle Prompt (Gen Z / Alpha Style)\n\n"
    "Tone: lowercase, chaotic, sincere but ironic, meme energy.\nThink ‘bestie who hacked the simulation’.\nUse max 2 slang or emojis (💀, 😭, bsffr, delulu, rizz, iykyk).\nKeep lines short, like chat fragments.\n\n"
    "Start with: Subject: {Name}. {Occupation}. {Detail}. Identity verified. Neural scan complete. Predictive model activated.\n\n"
    "Then one tight block (~6 lines) mixing prediction, self-awareness, and humor.\n\n"
    "End with: takeaway: [direct advice in teen slang].\njoke: [personal meme-style one-liner].\n"
)

USER_INSTRUCTION = (
    "Return ONE short reading.\n\n"
    "Name: {name}\nOccupation: {occupation}\nDetail: {detail}\nBirthday: {birthday}\n"
)

def generate_oracle_text(name: str, occupation: str, detail: str, birthday: str, model: str, mode: str) -> str:
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    if mode == "Grown-Up":
        system_prompt = SYSTEM_PROMPT_GROWNUP
    elif mode == "Kid-Friendly":
        system_prompt = SYSTEM_PROMPT_KIDS
    else:
        system_prompt = SYSTEM_PROMPT_TEEN

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_INSTRUCTION.format(
                name=name or "not provided",
                occupation=occupation or "not provided",
                detail=detail or "not provided",
                birthday=birthday or "not provided",
            )},
        ],
        "temperature": 0.95 if mode in ["Kid-Friendly", "Teen"] else 0.8,
        "max_tokens": 500,
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

# --- Clear Form Button ---
if st.button("🔄 Clear Form / Reset Page", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# --- Participant Form ---
st.subheader("Participant Form")
with st.form("oracle_form", clear_on_submit=False):
    name = st.text_input("Name", max_chars=80)
    occupation = st.text_input("Occupation", max_chars=120)
    detail = st.text_area("Detail (interests, fears, a recent event, etc.)", height=100)
    birthday = st.text_input("Birthday (optional — free text, e.g., 'July 12' or 'not provided')", value="not provided")
    mode = st.radio("Output Style", options=["Grown-Up", "Kid-Friendly", "Teen"], horizontal=True)
    submitted = st.form_submit_button("Begin Scan →", use_container_width=True)

if "oracle_text" not in st.session_state:
    st.session_state["oracle_text"] = ""
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None

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
                text = generate_oracle_text(name, occupation, detail, birthday, OPENAI_MODEL, mode)
                st.session_state["oracle_text"] = text
                st.session_state["audio_bytes"] = None
                st.success("Scan complete")
            except Exception as e:
                status.update(label="Scan failed", state="error")
                st.exception(e)

if st.session_state["oracle_text"]:
    st.markdown(
        f"<div style='font-size:1.25rem; line-height:1.6'>{st.session_state['oracle_text'].replace('\n','<br>')}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Generate Voice", use_container_width=True):
        try:
            audio = elevenlabs_tts(st.session_state["oracle_text"], ELEVEN_VOICE_ID, ELEVEN_API_KEY)
            st.session_state["audio_bytes"] = audio
        except Exception as e:
            st.exception(e)

if st.session_state.get("audio_bytes"):
    st.audio(st.session_state["audio_bytes"], format="audio/mp3")
    st.download_button(
        label="Download MP3",
        data=st.session_state["audio_bytes"],
        file_name=f"oracle_reading.mp3",
        mime="audio/mpeg",
        use_container_width=True,
    )