"""Simple configuration for emotion detection system."""

class Config:
    # Model Accuracies
    FACE_ACCURACY = "99.6%"
    VOICE_ACCURACY = "75.4%"
    
    # Fusion weights (face gets higher weight due to superior accuracy)
    FACE_WEIGHT = 0.7
    VOICE_WEIGHT = 0.3
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.5
    SPEAK_THRESHOLD = 0.6
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    BUFFER_SECONDS = 3
    
    # TTS settings for REAL-TIME
    SPEAK_INTERVAL = 1.0        # Reduced from 3.0 to 1.0 seconds
    SPEAK_THRESHOLD = 0.5       # Reduced from 0.6 to 0.5
    TTS_RATE = 180             # Faster speech
    TTS_VOLUME = 1.0           # Maximum volume
    
    # Emotions
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Voice model
    VOICE_MODEL = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'

