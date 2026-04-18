# CERTIS-AI-CHALLENGER--TEAM-AI-vengers
# 🔍 Local AI Crime Detection & Surveillance Agent

A fully local, privacy-first AI pipeline for analyzing surveillance video 
using on-device models — no cloud, no API keys required.

## 🧠 Models Used
- **LLaVA** (vision analysis) via Ollama
- **LLaMA 3.2** (reasoning & summarization) via Ollama
- **YAMNet + Librosa** (audio/acoustic event detection)

## ⚙️ Tech Stack
Python · Flask · OpenCV · LangChain · Whisper · pyttsx3

## 🚀 Features
- 3-frame extraction per video for fast analysis
- Real-time HTML dashboard on localhost:5000
- Officer-friendly one-sentence action summary
- Optional TTS (disable with `--no-tts`)
- Recursive UCF Crime Dataset support

## 📁 Dataset
[UCF Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/) — nested subfolders handled via `rglob`

## 🛠️ Setup
```bash
pip install -r requirements.txt
ollama pull llava
ollama pull llama3.2
python app.py
```

## 🔮 Future Enhancements
- RTSP live stream support
- Confidence scoring & alert tiering
- Exportable incident reports
- Mobile dashboard
