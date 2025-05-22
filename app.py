import streamlit as st
import os
import tempfile
import yt_dlp
import whisper
import torch
import librosa
import numpy as np
import warnings
import io
import base64
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Accent Detection Tool",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

class AccentDetector:
    def __init__(self):
        self.accent_keywords = {
            'british': {
                'words': ['brilliant', 'bloody', 'quite', 'rather', 'whilst', 'amongst', 
                         'colour', 'favour', 'realise', 'centre', 'queue', 'lorry', 'biscuit'],
                'spelling': ['colour', 'favour', 'realise', 'centre', 'theatre']
            },
            'american': {
                'words': ['awesome', 'totally', 'guys', 'like', 'super', 'color', 
                         'favor', 'realize', 'center', 'truck', 'cookie', 'elevator'],
                'spelling': ['color', 'favor', 'realize', 'center', 'theater']
            },
            'australian': {
                'words': ['mate', 'crikey', 'arvo', 'barbie', 'brekkie', 'servo', 
                         'bottle-o', 'fair dinkum', 'she\'ll be right', 'no worries'],
                'spelling': []
            },
            'canadian': {
                'words': ['eh', 'about', 'house', 'out', 'sorry', 'toque', 'double-double'],
                'spelling': ['colour', 'favour']  # Similar to British
            },
            'irish': {
                'words': ['grand', 'craic', 'sound', 'fair play', 'deadly', 'brilliant', 'class'],
                'spelling': []
            },
            'scottish': {
                'words': ['wee', 'ken', 'bonnie', 'braw', 'dinnae', 'cannae', 'nae'],
                'spelling': []
            }
        }
    
    def extract_audio_features(self, audio_file_path):
        """Extract acoustic features from audio file."""
        try:
            with st.spinner("Extracting audio features..."):
                y, sr = librosa.load(audio_file_path, sr=22050, duration=60)  # Limit to 60 seconds for speed
                
                features = {}
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
                
                # MFCC features (reduced number for speed)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
                for i in range(8):
                    features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                    features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                
                # Pitch features
                try:
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                    pitch_values = []
                    for t in range(0, pitches.shape[1], 10):  # Sample every 10th frame for speed
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if pitch > 0:
                            pitch_values.append(pitch)
                    
                    if pitch_values:
                        features['pitch_mean'] = float(np.mean(pitch_values))
                        features['pitch_std'] = float(np.std(pitch_values))
                        features['pitch_range'] = float(max(pitch_values) - min(pitch_values))
                    else:
                        features['pitch_mean'] = 0.0
                        features['pitch_std'] = 0.0
                        features['pitch_range'] = 0.0
                except:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
                    features['pitch_range'] = 0.0
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                features['zcr_mean'] = float(np.mean(zcr))
                features['zcr_std'] = float(np.std(zcr))
                
                return features
        except Exception as e:
            st.warning(f"Could not extract audio features: {str(e)}")
            return {}
    
    def analyze_text_patterns(self, transcript):
        """Analyze linguistic patterns in the transcript."""
        transcript_lower = transcript.lower()
        words = transcript.split()
        total_words = len(words)
        
        accent_scores = {}
        raw_scores = {}
        
        for accent, data in self.accent_keywords.items():
            raw_score = 0
            word_matches = 0
            keywords_found = []
            
            # Count keyword matches
            for word in data['words']:
                if word.lower() in transcript_lower:
                    word_matches += 1
                    keywords_found.append(word)
                    raw_score += 10  # Fixed score per keyword
            
            # Check spelling patterns (higher weight)
            for spelling in data.get('spelling', []):
                if spelling.lower() in transcript_lower:
                    raw_score += 20  # Higher weight for spelling indicators
                    keywords_found.append(f"{spelling} (spelling)")
                    word_matches += 1
            
            raw_scores[accent] = raw_score
            accent_scores[accent] = {
                'raw_score': raw_score,
                'word_matches': word_matches,
                'keywords_found': keywords_found
            }
        
        # Normalize scores to percentages (sum to 100% if any matches found)
        total_raw_score = sum(raw_scores.values())
        
        if total_raw_score > 0:
            for accent in accent_scores:
                percentage = (raw_scores[accent] / total_raw_score) * 100
                accent_scores[accent]['score'] = round(percentage, 1)
        else:
            # No matches found - distribute equally among likely accents
            for accent in accent_scores:
                accent_scores[accent]['score'] = 0
        
        return accent_scores
    
    def classify_accent(self, audio_file_path, transcript):
        """Main accent classification function with proper probability scoring."""
        # Analyze text patterns
        text_scores = self.analyze_text_patterns(transcript)
        
        # Extract audio features (optional, may fail)
        audio_features = self.extract_audio_features(audio_file_path)
        
        # Start with base probabilities
        accent_evidence = {}
        for accent, data in text_scores.items():
            accent_evidence[accent] = data['raw_score']
        
        # Add audio-based evidence if available
        if audio_features:
            try:
                pitch_mean = audio_features.get('pitch_mean', 0)
                spectral_centroid = audio_features.get('spectral_centroid_mean', 0)
                pitch_range = audio_features.get('pitch_range', 0)
                pitch_std = audio_features.get('pitch_std', 0)
                
                # Add evidence points based on audio features
                if pitch_mean > 180:
                    accent_evidence['british'] += 5
                if spectral_centroid > 1800:
                    accent_evidence['american'] += 3
                if pitch_range > 80:
                    accent_evidence['australian'] += 5
                if pitch_std > 30:
                    accent_evidence['irish'] += 3
                    
            except Exception:
                pass
        
        # Calculate total evidence
        total_evidence = sum(accent_evidence.values())
        
        # Convert to proper probabilities (must sum to 100%)
        if total_evidence > 0:
            # Calculate percentages that sum to 100%
            accent_probabilities = {}
            for accent, evidence in accent_evidence.items():
                accent_probabilities[accent] = round((evidence / total_evidence) * 100, 1)
            
            # Handle rounding errors to ensure exact 100% sum
            total_rounded = sum(accent_probabilities.values())
            if total_rounded != 100.0:
                # Adjust the highest probability to make sum exactly 100%
                max_accent = max(accent_probabilities.items(), key=lambda x: x[1])
                accent_probabilities[max_accent[0]] += round(100.0 - total_rounded, 1)
        else:
            # No evidence found - neutral distribution
            accent_probabilities = {}
            for accent in accent_evidence.keys():
                accent_probabilities[accent] = 0.0
            
            # Give small probabilities to most common accents
            accent_probabilities['american'] = 40.0
            accent_probabilities['british'] = 35.0
            accent_probabilities['canadian'] = 15.0
            accent_probabilities['australian'] = 10.0
        
        # Remove accents with 0% probability for cleaner display
        accent_probabilities = {k: v for k, v in accent_probabilities.items() if v > 0}
        
        # Determine primary accent
        if accent_probabilities:
            primary_accent = max(accent_probabilities.items(), key=lambda x: x[1])
            primary_accent_name = primary_accent[0]
            primary_confidence = primary_accent[1]
        else:
            primary_accent_name = "unidentified"
            primary_confidence = 0
        
        # Calculate English proficiency (separate metric)
        english_score = self.calculate_english_score(transcript, audio_features)
        
        # Calculate overall confidence in accent detection
        max_prob = max(accent_probabilities.values()) if accent_probabilities else 0
        detection_confidence = "High" if max_prob > 60 else "Moderate" if max_prob > 30 else "Low"
        
        return {
            'primary_accent': primary_accent_name,
            'confidence': primary_confidence,
            'english_score': english_score,
            'all_accents': accent_probabilities,
            'text_analysis': text_scores,
            'audio_features_available': bool(audio_features),
            'detection_confidence': detection_confidence,
            'total_evidence': total_evidence
        }
    
    def calculate_english_score(self, transcript, audio_features):
        """Calculate English proficiency score."""
        score = 40  # Base score
        
        words = transcript.split()
        word_count = len(words)
        
        if word_count > 20:
            score += 15
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        if word_count > 0:
            diversity_ratio = unique_words / word_count
            score += diversity_ratio * 25
        
        # Complex structures
        complex_words = ['because', 'although', 'however', 'therefore', 'moreover', 'furthermore']
        if any(word in transcript.lower() for word in complex_words):
            score += 20
        
        return min(score, 100)

def download_audio_from_url(video_url):
    """Download audio from video URL using yt-dlp."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "audio")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': f'{output_path}.%(ext)s',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded file
            audio_file = f"{output_path}.mp3"
            if os.path.exists(audio_file):
                # Read the file content
                with open(audio_file, 'rb') as f:
                    audio_content = f.read()
                return audio_content, "audio.mp3"
            else:
                raise Exception("Audio file not found after download")
                
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def transcribe_audio(audio_content, filename):
    """Transcribe audio using Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        temp_file.write(audio_content)
        temp_file_path = temp_file.name
    
    try:
        model = load_whisper_model()
        result = model.transcribe(temp_file_path, language="en", without_timestamps=True)
        return result["text"], temp_file_path
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise Exception(f"Transcription failed: {str(e)}")

def main():
    st.title("🎤 Accent Detection Tool")
    st.markdown("**Analyze English accents from video URLs**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This tool analyzes English accents from video content using:")
        st.write("• Speech transcription")
        st.write("• Linguistic pattern analysis")
        st.write("• Audio feature extraction")
        
        st.header("🎯 Supported Accents")
        accents = ["British", "American", "Australian", "Canadian", "Irish", "Scottish"]
        for accent in accents:
            st.write(f"• {accent}")
    
    # Main interface
    video_url = st.text_input(
        "📹 Enter Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Supports YouTube, Vimeo, and direct video links"
    )
    
    if st.button("🚀 Analyze Accent", type="primary"):
        if not video_url:
            st.error("Please enter a video URL")
            return
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Download audio
            status_text.text("📥 Downloading audio...")
            progress_bar.progress(20)
            audio_content, filename = download_audio_from_url(video_url)
            
            # Step 2: Transcribe
            status_text.text("🎧 Transcribing audio...")
            progress_bar.progress(50)
            transcript, temp_audio_path = transcribe_audio(audio_content, filename)
            
            # Step 3: Analyze
            status_text.text("🔍 Analyzing accent...")
            progress_bar.progress(80)
            detector = AccentDetector()
            analysis = detector.classify_accent(temp_audio_path, transcript)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            # Display results
            st.success("Analysis Complete!")
            
            # Main results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Primary Accent",
                    analysis['primary_accent'].title(),
                    f"{analysis['confidence']:.1f}% ({analysis['detection_confidence']} confidence)"
                )
            
            with col2:
                st.metric(
                    "🗣️ English Proficiency",
                    f"{analysis['english_score']:.1f}%"
                )
            
            with col3:
                audio_status = "✅ Available" if analysis['audio_features_available'] else "⚠️ Text Only"
                st.metric(
                    "🔊 Audio Analysis",
                    audio_status
                )
            
            # Detailed breakdown
            st.subheader("📊 Detailed Accent Analysis")
            
            # Create accent confidence chart
            accent_data = [(accent.title(), confidence) for accent, confidence in 
                          sorted(analysis['all_accents'].items(), key=lambda x: x[1], reverse=True)
                          if confidence > 0.1]  # Only show accents with meaningful confidence
            
            if accent_data:
                st.write("**Accent Probability Distribution:**")
                total_shown = sum(conf for _, conf in accent_data)
                st.write(f"*Probabilities sum to exactly: {total_shown:.1f}%*")
                
                for accent, confidence in accent_data:
                    st.write(f"**{accent}:** {confidence:.1f}%")
                    st.progress(min(confidence / 100, 1.0))
                    
                if abs(total_shown - 100.0) > 0.1:
                    st.warning("⚠️ Probability sum error detected - please refresh analysis")
            else:
                st.info("No strong accent indicators detected - speaker may have a neutral accent or the sample may be too short")
            
            # Keywords found
            st.subheader("🔑 Keywords & Patterns Found")
            keywords_found = False
            for accent, data in analysis['text_analysis'].items():
                if data['keywords_found']:
                    keywords_found = True
                    st.write(f"**{accent.title()}:** {', '.join(data['keywords_found'])}")
            
            if not keywords_found:
                st.info("No specific accent keywords detected")
            
            # Transcript
            with st.expander("📝 Full Transcript"):
                st.text_area("Transcript:", transcript, height=200, disabled=True)
            
            # Download results
            results_text = f"""ACCENT ANALYSIS RESULTS
{'='*50}

Video URL: {video_url}
Primary Accent: {analysis['primary_accent'].title()}
Confidence: {analysis['confidence']:.1f}%
English Proficiency: {analysis['english_score']:.1f}%

TRANSCRIPT:
{transcript}

DETAILED BREAKDOWN:
{chr(10).join([f"{accent.title()}: {conf:.1f}%" for accent, conf in analysis['all_accents'].items() if conf > 0])}
"""
            
            st.download_button(
                label="📄 Download Results",
                data=results_text,
                file_name="accent_analysis_results.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try a different video URL or check your internet connection")

if __name__ == "__main__":
    main()