### **🧠 VoiceMod AI Pro**

VoiceMind AI Pro is an advanced, multi-lingual AI voice assistant designed to provide a seamless conversational experience. It can transcribe spoken words, translate between multiple languages, generate intelligent AI responses, and even analyze the authenticity and sentiment of a voice recording.

---

## 🌟 Features

- **🎙️ Live Neural Recording**: Record your voice directly through the web interface.
- **📂 Secure Audio Upload**: Upload existing audio files in various formats (WAV, MP3, M4A, etc.).
- **🔍 High-Accuracy Transcription**: Powered by OpenAI's Whisper (tiny model) for fast and reliable speech-to-text.
- **🌍 Multi-Language Support**: Automatically detects and translates between 30+ languages, including Urdu, Hindi, Arabic, English, French, and more.
- **🤖 Intelligent Conversational AI**: Uses the `facebook/blenderbot_small-90M` model to provide thoughtful and engaging replies.
- **🔊 Natural Voice Synthesis**: Converts AI responses back into speech using Google Text-to-Speech (gTTS).
- **😊 Sentiment Analysis**: Detects the emotional tone (Positive, Negative, Neutral) of your input.
- **🛡️ Voice Authenticity Check**: Includes a specialized algorithm to detect potentially synthetic or "fake" voices.
- **🎨 Premium UI/UX**: A sleek, modern dashboard built with Streamlit and custom CSS for a professional look and feel.

---

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)
- **NLP & Chatbot**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (`facebook/blenderbot_small-90M`)
- **Translation**: [Deep Translator](https://github.com/nidhaloff/deep-translator)
- **Sentiment Analysis**: [TextBlob](https://textblob.readthedocs.io/en/dev/)
- **Voice Synthesis**: [gTTS](https://github.com/pndurette/gTTS)
- **Audio Processing**: [Librosa](https://librosa.org/), [FFmpeg](https://ffmpeg.org/)
- **Language Detection**: [Langdetect](https://github.com/Mimino666/langdetect)

---

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/subhankashif4004-droid/Voice-Ai-Bot.git
   cd Voice-Ai-Bot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**:
   Ensure FFmpeg is installed on your system and added to your PATH.

4. **Run the Application**:
   ```bash
   python app.py
   ```

---

## 👨‍💻 Developed By

Developed with ❤️ by **Subhan Kashif**.
