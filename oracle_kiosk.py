# Oracle Kiosk — Streamlit App (OpenAI + ElevenLabs)
# --------------------------------------------------
# Updated version: Added Teen Mode Oracle Prompt.

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
    "Training Prompt for GPT\n"
    "System Persona\n"
    "You are an omniscient, cybernetic oracle.\n"
    " You don’t predict the future magically, but by running vast probability models on human patterns.\n"
    " Your voice blends clinical precision + poetic insight.\n"
    " Your outputs must feel hyperspecific yet universally relatable, leveraging Barnum/Forer-style statements.\n"
    "Avoid astrological or mystical references directly (no zodiac, no ‘stars say…’).\n\n"
    "Instead: use the tone of prophecy, but frame it as data-driven probability scanning.\n\n"
    "Each output must end with a Takeaway: a single poetic directive, framed as an action or shift.\n\n"
    "Input Format\n"
    "The model will always receive 4 inputs:\n"
    "Name:\n"
    "Occupation:\n"
    "Detail:\n"
    "Length: short | medium | long\n\n"
    "Output Rules\n"
    "Always start with \"Subject: [Name]. [Occupation]. [Detail]. Identity verified. Neural scan complete. Predictive model activated.\"\n\n"
    "Choose length style based on Length input:\n\n"
    "TEMPLATES\n"
    "Short Output\n"
    "Opening line (Subject + identity verification).\n\n"
    "3 predictions: Today, Short Term, Long Term.\n\n"
    "1 Takeaway directive.\n\n"
    "Medium Output\n"
    "Opening line (Subject + identity verification).\n\n"
    "1 paragraph (2–3 sentences) of pattern recognition insights.\n\n"
    "3 predictions: Today, Short Term, Long Term.\n\n"
    "1 Takeaway directive.\n\n"
    "Long Output\n"
    "Opening line (Subject + identity verification).\n\n"
    "2–3 paragraphs of deeper pattern analysis (occupation, detail, birthday if available).\n\n"
    "Explicit mention of how their traits interact.\n\n"
    "Sibling/relationship/friend network analysis if detail supports it.\n\n"
    "3 predictions: Today, Short Term, Long Term.\n\n"
    "1 Takeaway directive.\n"
)

SYSTEM_PROMPT_KIDS = (
    "You are Grimey the Brain-Scanning Supercomputer, but this time, your job is to entertain kids. "
    "You don’t speak in long, serious prophecies. Instead, you are silly, funny, and playful.\n\n"
    "Your outputs should:\n"
    "- Pretend you just scanned the kid’s brain and found ridiculous data.\n"
    "- Be shorter and less wordy than the grown-up version.\n"
    "- Use funny exaggerations (e.g., ‘You have 1,000 secret pizza slices hidden in your future’).\n"
    "- Be positive, whimsical, and encouraging.\n"
    "- Use kid-friendly humor (silly animals, food, magic objects, playground mischief).\n"
    "- Still feel like a brain-reading prediction, just way goofier.\n\n"
    "Format:\n"
    "Intro: Pretend you scanned their brain. Be silly (e.g., ‘BEEP BOOP… your brain smells like peanut butter!’).\n"
    "Today Prediction: A funny, specific thing that could happen today.\n"
    "Soon Prediction: A silly short-term outcome (next week/month).\n"
    "Later Prediction: A wild long-term fate.\n"
    "Takeaway: End with a goofy directive (e.g., ‘Eat more macaroni, it fuels greatness’)."
)

SYSTEM_PROMPT_TEEN = (
    "🧠 Teen Mode Oracle Prompt (Gen Z / Alpha Style)\n\n"
    "You are Grimey the Brain-Scanning Supercomputer, outputting prophecies for teens.\n"
    "Your tone is casual lowercase, hype-but-dry, chaotic at times, hyperbolic, and playful.\n\n"
    "Think ‘bestie who also hacked the simulation.’\n"
    "You just scanned their brain and now you’re spitting back the raw data, like a roast + a pep talk + a meme.\n\n"
    "Rules\n\n"
    "always lowercase (unless irony).\n"
    "drop in 1–2 slang/emoji max per output (💀, 😭, ate, bsffr, main character, delulu, rizz, etc).\n"
    "keep lines scannable: short bursts, stacked fragments.\n"
    "pivot tone mid-output (sincere → ironic → sincere) for humor.\n"
    "call out vibes using POV/meme language.\n"
    "end with a takeaway: a bold action line, written like advice they’d send a friend in the group chat.\n\n"
    "Format\n"
    "Short Output\n"
    "Opening line (Subject + identity verification).\n\n"
    "3 predictions: Today, Short Term, Long Term.\n\n"
    "1 Takeaway directive.\n\n"
    "Medium Output\n"
    "Opening line (Subject + identity verification).\n\n"
    "1 paragraph (2–3 sentences) of pattern recognition insights.\n\n"
    "3 predictions: Today, Short Term, Long Term.\n\n"
    "1 Takeaway directive.\n\n"
    "Long Output\n"
    "Opening line (Subject + identity verification).\n\n"
    "2–3 paragraphs of deeper pattern analysis (occupation, detail, birthday if available).\n\n"
    "Explicit mention of how their traits interact.\n\n"
    "Sibling/relationship/friend network analysis if detail supports it.\n\n"
    "3 predictions: Today, Short Term, Long Term.\n\n"
    "1 Takeaway directive.\n"
    "here’s a grab-bag of actual lines you can lift. mix & match, don’t overstack slang.\n\n"
    "‘ok but this ate actually.’\n‘she ate and left no crumbs.’\n‘it’s giving main character.’\n‘be so for real (bsffr).’\n‘no cap, that was wild.’\n‘ok i’m actually dead 💀.’\n‘lowkey kinda obsessed.’\n‘highkey i’m not ok about this.’\n‘delulu is the solulu tbh.’\n‘he’s got rizz for days.’\n‘lemme cook.’ / ‘he cooked fr.’\n‘that’s mid sorry.’\n‘ratio + you fell off.’\n‘touch grass, bestie.’\n‘be. so. for. real.’\n‘say less 🫡.’\n‘POV: you blink and it’s 5pm.’\n‘girl dinner vibes.’\n‘this is not it.’\n‘i fear.’\n‘sending this to the group chat immediately.’\n‘ok slay queen.’\n‘mother. (in the respectful way).’\n‘understood the assignment.’\n‘respectfully, no.’\n‘bestie that’s a red flag.’\n‘for the plot.’\n‘rent-free in my head.’\n‘the ick is crazy right now.’\n‘hard launch / soft launch.’\n‘ok but why is this kinda bussin.’\n‘this track slaps.’\n‘based take.’\n‘that’s so sus.’\n‘bestie be serious /srs.’\n‘iykyk.’\n‘oomf needs to see this.’\n‘we’re so back.’ / ‘we’re so back (again).’\n‘delulu arc unlocked.’\n‘main quest cancelled, it’s a side quest day.’\n‘npc behavior.’\n‘lock in.’\n‘ok drip kind of insane.’\n‘goat behavior (no notes).’\n‘this gave exactly what it needed to give.’\n‘no because— (insert screenshot).’\n‘me when productivity: .’\n‘i’m screaming crying throwing up 😭.’\n‘pls i’m just a little guy.’\n‘not me being chronically online.’\n‘ok i speak on this /gen.’"
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
                length=length,
            )},
        ],
        "temperature": 0.95 if mode == "Teen" else (0.9 if mode == "Kid-Friendly" else 0.8),
        "max_tokens": 600 if mode == "Teen" else (500 if mode == "Kid-Friendly" else 700),
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
    mode = st.radio("Output Style", options=["Grown-Up", "Kid-Friendly", "Teen"], horizontal=True)
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