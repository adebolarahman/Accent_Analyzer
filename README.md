# Accent_Analyzer

# 🎤 Accent Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://accentanalyzer-nbwygpiyzopo5mkp3geaxl.streamlit.app/)

An intelligent tool that analyzes English accents from video content, providing automated assessment capabilities for hiring and language evaluation purposes.

## 🌟 Features

- **Video Input**: Supports YouTube, Vimeo, and direct video links
- **Speech Transcription**: Uses faster-whisper for accurate speech-to-text conversion
- **Accent Classification**: Identifies British, American, Australian, Canadian, Irish, and Scottish accents
- **Audio Analysis**: Extracts spectral features, MFCC coefficients, and pitch characteristics
- **English Proficiency**: Provides confidence scores and proficiency percentages
- **Real-time Processing**: Live progress tracking with step-by-step feedback
- **Exportable Results**: Download detailed analysis reports

## 🚀 Live Demo

Try the app: [Accent Analyzer](https://accentanalyzer-nbwygpiyzopo5mkp3geaxl.streamlit.app/)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Speech Recognition**: faster-whisper
- **Audio Processing**: librosa, NumPy, SciPy
- **Video Processing**: yt-dlp
- **Deployment**: Streamlit Cloud

## 🏃‍♂️ Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adebolarahman/Accent_Analyzer.git
   cd Accent_Analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (for audio processing)**
   ```bash
   # Windows
   # Download FFmpeg from https://ffmpeg.org/download.html
   
   # macOS
   brew install ffmpeg
   
   # Linux
   sudo apt update
   sudo apt install ffmpeg libsndfile1 libsox-dev
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 How It Works

1. **Video Input**: Enter a public video URL
2. **Audio Extraction**: Download and extract audio from the video
3. **Speech Transcription**: Convert speech to text using faster-whisper
4. **Linguistic Analysis**: Analyze text patterns and vocabulary markers
5. **Audio Feature Extraction**: Process spectral and pitch characteristics
6. **Classification**: Combine text and audio features for accent identification
7. **Results**: Display confidence scores, probabilities, and detailed analysis

## 🎯 Supported Accents

- 🇬🇧 British
- 🇺🇸 American  
- 🇦🇺 Australian
- 🇨🇦 Canadian
- 🇮🇪 Irish
- 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish

## 📈 Performance

- **Accuracy**: ~70-80% for text-based analysis
- **Processing Speed**: 15-30 seconds for 60-second videos
- **Video Length**: Optimized for videos under 5 minutes
- **Cost**: Zero transcription costs (local processing)

## 🔮 Future Enhancements

- **Improved Accuracy**: Fine-tuning with larger datasets (target: 90-95%)
- **Additional Accents**: Indian, South African, and other English variants
- **API Integration**: OpenAI Whisper API for enterprise deployment
- **Machine Learning**: Distance metrics and supervised learning models
- **Batch Processing**: Support for multiple videos simultaneously

## 🚀 Production Deployment

For enterprise use, consider:
- **OpenAI Whisper API**: Superior accuracy and faster processing
- **Cloud Infrastructure**: Scalable processing with containerization
- **Database Integration**: Store analysis results and user data
- **Authentication**: User management and access controls

## 📁 Project Structure

```
accent_analyzer/
├── main.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Adebola Rahman**

- GitHub: [@adebolarahman](https://github.com/adebolarahman)
- LinkedIn: [Connect with me](https://linkedin.com/in/adebolarahman)

## 🙏 Acknowledgments

- OpenAI for the Whisper model architecture
- Streamlit team for the excellent web framework
- librosa developers for comprehensive audio analysis tools

---

⭐ Star this repo if you found it helpful!
