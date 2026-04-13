import sys
# Bug fix: Block the incompatible upb module so protobuf falls back to pure python on Python 3.14
sys.modules["google._upb"] = None
sys.modules["google._upb._message"] = None

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import shutil

# Wrap the main application logic in a function to prevent premature execution
def run_app():
    import whisper
    from core.utils import (
        detect_language, translate_to_english, translate_back,
        generate_ai_response, analyze_sentiment, fake_voice_detection,
        safe_lang_for_gtts, get_language_name, get_language_code
    )
    import core.audio_utils as audio_utils
    from gtts import gTTS
    import tempfile
    import threading

    st.set_page_config(page_title="VoiceMind AI Pro", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

    with open("assets/css/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown("""
        <style>
            [data-testid="collapsedControl"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def get_whisper_lock():
        return threading.Lock()

    def check_ffmpeg():
        return shutil.which("ffmpeg") is not None

    if not check_ffmpeg():
        st.error("❌ `ffmpeg` not found. Please install it or add it to your PATH for audio processing.")
        st.info("💡 You can download it from https://ffmpeg.org/download.html")
        st.stop()

    @st.cache_resource
    def load_whisper():
        with st.spinner("📦 Loading Whisper model..."):
            return whisper.load_model("tiny")

    @st.cache_resource
    def load_chatbot():
        with st.spinner("📦 Loading Neural Chatbot..."):
            from transformers import pipeline
            return pipeline("text2text-generation", model="facebook/blenderbot_small-90M")

    whisper_lock = get_whisper_lock()
    model = load_whisper()
    chatbot = load_chatbot()

    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "result" not in st.session_state:
        st.session_state.result = None

    left_panel, main_panel, spacer = st.columns([1, 2.3, 0.5], gap="large")

    with left_panel:
        st.markdown('<div class="upload-card" style="position: sticky; top: 20px;">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Control Panel")
        st.markdown("<p style='color: #8892b0; font-size: 0.9rem;'>Configure your voice assistant settings below.</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Audio Source")
        st.info("Listening for voice or file uploads.\n\n🌍 **Supports: Urdu, Hindi, Arabic, English, and 30+ languages!**")
        st.markdown("---")
        st.markdown("#### Processing Mode")
        st.success("🧠 High-Speed Neural Pipeline Active")
        st.markdown("---")
        st.markdown("#### 🗣️ AI Response Settings")
        target_lang_name = st.selectbox(
            "Choose Response Language", 
            ["Auto-Reply in my Language", "English", "Urdu", "Hindi", "French", "Spanish", "German", "Arabic", "Japanese", "Russian"],
            index=0
        )
        st.markdown("---")
        st.markdown("<p style='font-size: 0.8rem; color: #555; text-align: center; margin-top: 50px;'>VoiceMind AI Pro v2.0</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with main_panel:
        st.markdown("<h1 class='main-title'>🧠 VoiceMind AI Pro</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Speak naturally or upload an audio file. Our AI will transcribe, translate, understand, and reply back to you.</p>", unsafe_allow_html=True)
        st.divider()

        st.markdown("### 🔴 Live Neural Recording")
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        
        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False

        st.markdown('<div class="record-controls">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🟢 Start Recording", use_container_width=True, type="primary", disabled=st.session_state.is_recording):
                success = audio_utils.start_recording()
                if success:
                    st.session_state.is_recording = True
                    st.rerun()
                else:
                    st.error("❌ Failed to start recording. Check your microphone settings.")
                
        with col2:
            if st.button("🔴 Stop Recording", use_container_width=True, type="primary", disabled=not st.session_state.is_recording):
                st.session_state.is_recording = False
                with st.spinner("💾 Saving recording..."):
                    audio_path = audio_utils.stop_recording()
                if audio_path:
                    st.session_state.audio_path = audio_path
                    st.session_state.result = None
                    st.success("✅ Recording complete!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.is_recording:
            st.success("🟢 Recording in progress... Press 'Stop Recording' when done.")
            
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center; color: #3b82f6; margin: 20px 0; font-weight: 800;'>— OR —</h3>", unsafe_allow_html=True)
        
        st.markdown("### 📂 Secure Audio Upload")
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        audio_file = st.file_uploader("", type=["wav", "mp3", "m4a", "mpeg", "mp4", "ogg", "flac"], label_visibility="collapsed")
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                tmp.write(audio_file.read())
                st.session_state.audio_path = tmp.name
            st.session_state.result = None
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        if st.session_state.audio_path and st.session_state.result is None:
            audio_path = st.session_state.audio_path
            st.audio(audio_path)

            try:
                if os.path.getsize(audio_path) == 0:
                    st.warning("⚠️ The recorded or uploaded audio is empty. Please try again.")
                    st.stop()

                with st.spinner("🔍 Transcribing..."):
                    with whisper_lock:
                        result = model.transcribe(audio_path, fp16=False)
                    original_text = result["text"].strip()

                if not original_text:
                    st.warning("⚠️ No speech detected.")
                    st.stop()

                lang = detect_language(original_text)
                english_text = translate_to_english(original_text, lang)

                with st.spinner("🧠 Generating AI response..."):
                    ai_response_en = generate_ai_response(chatbot, english_text)
                    if not ai_response_en or ai_response_en.strip() == "":
                        ai_response_en = "I am sorry, I could not generate a proper response."

                target_code = lang if target_lang_name == "Auto-Reply in my Language" else get_language_code(target_lang_name)
                final_response = translate_back(ai_response_en, target_code)
                if not final_response or final_response.strip() == "":
                    final_response = "Sorry, translation failed."

                with st.spinner("🔊 Generating voice reply..."):
                    gtts_lang = safe_lang_for_gtts(target_code)
                    tts = gTTS(text=final_response, lang=gtts_lang)
                    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    tts.save(tts_path)

                with st.spinner("📊 Finalizing analysis..."):
                    mood, score = analyze_sentiment(english_text)
                    fake_result = fake_voice_detection(audio_path)

                st.session_state.result = {
                    "original_text": original_text,
                    "lang": lang,
                    "english_text": english_text,
                    "ai_response_en": ai_response_en,
                    "final_response": final_response,
                    "mood": mood,
                    "score": score,
                    "fake_result": fake_result,
                    "tts_path": tts_path,
                }

            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.stop()

        if st.session_state.result:
            r = st.session_state.result
            st.divider()

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader("📝 You Said")
            st.write(r["original_text"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader("🌍 Detected Language")
            st.write(get_language_name(r["lang"]))
            st.markdown('</div>', unsafe_allow_html=True)

            if r["lang"] != "en":
                st.markdown('<div class="box">', unsafe_allow_html=True)
                st.subheader("🔄 In English")
                st.write(r["english_text"])
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader("🤖 AI Reply")
            st.write(r["final_response"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader("😊 Sentiment")
            st.write(f"{r['mood']}  (score: {r['score']})")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.subheader("🔍 Voice Authenticity")
            st.write(r["fake_result"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("🔊 AI Voice Reply")
            st.markdown('<div class="ai-audio-player">', unsafe_allow_html=True)
            st.audio(r["tts_path"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.divider()
            if st.button("🔄 Start Over"):
                st.session_state.audio_path = None
                st.session_state.result = None
                st.rerun()

    st.markdown('<div class="footer">🚀 Developed by Subhan</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    import subprocess
    try:
        # Check if we are already running inside Streamlit
        instance = st.runtime.get_instance()
    except RuntimeError:
        instance = None

    if instance is None:
        # Bare mode: Ultimate Monkey-Patch for Python 3.14
        # We manually patch the child process via command line before it imports Streamlit
        cmd = [
            sys.executable, "-c",
            "import sys, os; "
            "sys.modules['google._upb']=None; "
            "sys.modules['google._upb._message']=None; "
            "os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'; "
            "from streamlit.web.cli import main; "
            "sys.argv=['streamlit', 'run', r'" + __file__ + "']; "
            "main()"
        ]
        subprocess.run(cmd)
    else:
        # Streamlit mode: Run the actual application
        run_app()
else:
    # Being imported by Streamlit runner
    run_app()