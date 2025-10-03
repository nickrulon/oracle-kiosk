# Oracle Kiosk — Streamlit App (OpenAI + ElevenLabs)
# --------------------------------------------------
# One-file Streamlit app that:
# 1) Collects a participant's info via a form (Name, Occupation, Detail, optional Birthday, Length)
# 2) Calls OpenAI to generate a cybernetic-oracle reading using your provided templates
# 3) Lets an admin review/approve the text
# 4) Sends approved text to ElevenLabs TTS and plays the audio
#
# Setup
# -----
# 1) Create a virtual env (optional) and install deps:
#    pip install streamlit openai==1.* requests python-dotenv
# 2) Put your keys in a .env file (same folder) or export as env vars:
#    OPENAI_API_KEY=sk-...
#    ELEVENLABS_API_KEY=...
#    ELEVENLABS_VOICE_ID=YOUR_VOICE_ID
# 3) Run the app:
#    streamlit run oracle_kiosk.py
#
# Tips for show mode
# ------------------
# • Open two browser windows to the app:
#   - Admin View (toggle off "Audience Mode") for controls/regenerate/approve
#   - Audience View (toggle on "Audience Mode") to show only scanning + final text/audio
# • Use an HDMI/USB‑C output to mirror the Audience View to your big display.

import os
import time
import io
import requests
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Optional .env loader (safe no-op if file doesn't exist)
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or st.secrets.get("ELEVENLABS_VOICE_ID", "")

# Model name: pick a fast, cost-effective chat model. Adjust if you prefer another.
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

# Basic guardrails
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not set. Add it to your environment or a .env file.")
if not ELEVEN_API_KEY:
    st.warning("ELEVENLABS_API_KEY not set. Add it to your environment or a .env file.")
if not ELEVEN_VOICE_ID:
    st.info("Tip: set ELEVENLABS_VOICE_ID for your custom voice. You can still paste a Voice ID below.")

# --- Prompt Content (from your Training Prompt) ---
SYSTEM_PROMPT = (
    "You are an omniscient, cybernetic oracle. "
    "You don’t predict the future magically, but by running vast probability models on human patterns. "
    "Your voice blends clinical precision + poetic insight. "
    "Your outputs must feel hyperspecific yet universally relatable, leveraging Barnum/Forer-style statements. "
    "Avoid astrological or mystical references directly (no zodiac, no ‘stars say…’). "
    "Use the tone of prophecy, but frame it as data-driven probability scanning. "
    "Each output must end with a Takeaway: a single poetic directive, framed as an action or shift.\n\n"
    "Always start with: \"Subject: [Name]. [Occupation]. [Detail]. Identity verified. Neural scan complete. Predictive model activated.\"\n"
    "Select the correct template (short, medium, or long) based on the provided Length input.\n"
    "Incorporate Name, Occupation, Detail, and optional Birthday (if given, weave seasonal/monthly archetypes subtly).\n"
    "Return exactly one oracle output."
)

TEMPLATE_SHORT = (
    "Subject: {Name}. {Occupation}. {Detail}. Identity verified. Neural scan complete. Predictive model activated.\n\n"
    "Today: [single immediate signal].\n"
    "Short Term: [near-future pivot/advice].\n"
    "Long Term: [longer arc outcome with a condition].\n"
    "Takeaway: \"[concise poetic directive].\""
)

TEMPLATE_MEDIUM = (
    "Subject: {Name}. {Occupation}. {Detail}. Identity verified. Neural scan complete. Predictive model activated.\n\n"
    "Pattern recognition reveals [2–3 dominant clusters or traits linked to their occupation/detail]. "
    "These are not random — they’re the key nodes in the probability graph of your choices.\n\n"
    "Today: [signal or opportunity].\n"
    "Short Term: [pivot or warning in 2–4 weeks].\n"
    "Long Term: [longer trajectory insight].\n"
    "Takeaway: \"[poetic directive].\""
)

TEMPLATE_LONG = (
    "Subject: {Name}. {Occupation}. {Detail}. Identity verified. Neural scan complete. Predictive model activated.\n\n"
    "[Paragraph 1: highlight 2–3 key life patterns linked to occupation and detail. Frame as probability clusters or behavior loops.]\n"
    "[Paragraph 2: describe differentiator or hidden strength. Connect detail (siblings, hobbies, quirks) into broader trajectory. "
    "If birthday provided, weave in seasonal archetype without naming zodiac.]\n"
    "[Optional Paragraph 3: extend into network/relationships, showing how they shape near-future decisions.]\n\n"
    "Today: [immediate small signal].\n"
    "Short Term: [near-future pivot/advice].\n"
    "Long Term: [longer arc trajectory].\n"
    "Takeaway: \"[poetic directive with subtle ‘system output’ flavor].\""
)

# Slightly opinionated helper to build the user content fed to the model
USER_INSTRUCTION = (
    "When the user provides input in the following format, return one completed oracle output "
    "strictly following the templates above.\n\n"
    "Name: {name}\n"
    "Occupation: {occupation}\n"
    "Detail: {detail}\n"
    "Birthday: {birthday}\n"
    "Length: {length}\n"
)

# --- OpenAI Chat API (python SDK v1 style via requests fallback) ---

def generate_oracle_text(name: str, occupation: str, detail: str, birthday: str, length: str, model: str) -> str:
    """Call OpenAI Chat Completions to produce the oracle output."""
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_INSTRUCTION.format(
                    name=name or "not provided",
                    occupation=occupation or "not provided",
                    detail=detail or "not provided",
                    birthday=birthday or "not provided",
                    length=length,
                ),
            },
        ],
        "temperature": 0.8,
        "max_tokens": 700,
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# --- ElevenLabs TTS ---

def elevenlabs_tts(text: str, voice_id: str, api_key: str) -> bytes:
    """Send text to ElevenLabs and return raw audio bytes (mp3)."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content

# --- UI ---
st.set_page_config(page_title="Oracle Kiosk", page_icon="🔮", layout="wide")
st.title("🔮 Oracle Kiosk — Cybernetic Reading → Voice")

colL, colR = st.columns([3, 2])
with colR:
    st.subheader("Settings")
    audience_mode = st.toggle("Audience Mode (clean stage view)", value=False, help="Hides admin controls & shows big output.")
    model_name = st.text_input("OpenAI Model", value=OPENAI_MODEL, help="e.g., gpt-4o-mini, gpt-4o, gpt-4.1-mini, etc.")
    voice_id = st.text_input("ElevenLabs Voice ID", value=ELEVEN_VOICE_ID)
    auto_tts = st.toggle("Auto-play after approval", value=True)

with colL:
    st.subheader("Participant Form")
    with st.form("oracle_form", clear_on_submit=False):
        name = st.text_input("Name", max_chars=80)
        occupation = st.text_input("Occupation", max_chars=120)
        detail = st.text_area("Detail (interests, fears, a recent event, etc.)", height=100)
        birthday = st.text_input("Birthday (optional — free text, e.g., 'July 12' or 'not provided')", value="not provided")
        length = st.radio("Length", options=["short", "medium", "long"], horizontal=True)
        submitted = st.form_submit_button("Begin Scan →", use_container_width=True)

# State for latest generated text & audio
if "oracle_text" not in st.session_state:
    st.session_state["oracle_text"] = ""
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None
if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = 0.0

# Handle submission → generate text
if submitted:
    if not all([OPENAI_API_KEY]) or not model_name:
        st.error("Missing OpenAI credentials or model name.")
    else:
        with st.status("Neural scan in progress…", expanded=True) as status:
            st.write("Uploading neural data…")
            time.sleep(0.6)
            st.write("Pattern graph building…")
            time.sleep(0.6)
            st.write("Probability mesh converging…")
            t0 = time.time()
            try:
                text = generate_oracle_text(name, occupation, detail, birthday, length, model_name)
                st.session_state["oracle_text"] = text
                st.session_state["audio_bytes"] = None  # reset any previous audio
                st.session_state["scan_time"] = time.time() - t0
                status.update(label="Scan complete", state="complete")
            except Exception as e:
                status.update(label="Scan failed", state="error")
                st.exception(e)

# Stage / Admin panels
if audience_mode:
    st.header("Output")
    if st.session_state["oracle_text"]:
        st.markdown(
            f"<div style='font-size:1.25rem; line-height:1.6'>{st.session_state['oracle_text'].replace('\n','<br>')}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Awaiting a scan…")
    if st.session_state["audio_bytes"]:
        st.audio(st.session_state["audio_bytes"], format="audio/mp3")
else:
    st.subheader("Admin Review")
    if st.session_state["oracle_text"]:
        st.caption(f"Model: {model_name} · Generated in {st.session_state['scan_time']:.1f}s")
        st.text_area("Oracle Text (review/edit before voice)", value=st.session_state["oracle_text"], height=300, key="oracle_text_edit")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("Regenerate", use_container_width=True):
                # Re-run generation with same inputs
                try:
                    t0 = time.time()
                    text = generate_oracle_text(name, occupation, detail, birthday, length, model_name)
                    st.session_state["oracle_text"] = text
                    st.session_state["oracle_text_edit"] = text
                    st.session_state["audio_bytes"] = None
                    st.session_state["scan_time"] = time.time() - t0
                    st.success("Regenerated.")
                except Exception as e:
                    st.exception(e)
        with c2:
            approve = st.button("Approve → Generate Voice", use_container_width=True)
        if approve:
            if not ELEVEN_API_KEY and not voice_id:
                st.error("Missing ElevenLabs API key or Voice ID.")
            else:
                try:
                    text_for_tts = st.session_state.get("oracle_text_edit") or st.session_state["oracle_text"]
                    audio = elevenlabs_tts(text_for_tts, voice_id or ELEVEN_VOICE_ID, ELEVEN_API_KEY)
                    st.session_state["audio_bytes"] = audio
                    st.success("Voice generated.")
                except Exception as e:
                    st.exception(e)
        if st.session_state["audio_bytes"]:
            st.audio(st.session_state["audio_bytes"], format="audio/mp3")
            st.download_button(
                label="Download MP3",
                data=st.session_state["audio_bytes"],
                file_name=f"oracle_reading_{int(time.time())}.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )
            if auto_tts:
                st.balloons()
    else:
        st.info("Submit the form to generate an oracle output for review.")

st.markdown("""
---
**Security note:** Streamlit runs in your environment. Your API keys remain local if you only run this on your machine. If you deploy to cloud, set keys as environment variables and never hard‑code them.
""")

