"""Real-time audio capture for voice emotion detection."""

import numpy as np
import threading
import queue
import time
from config import Config

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ SoundDevice loaded successfully")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  SoundDevice not available")

class AudioCapture:
    """Real-time audio capture and processing."""
    
    def __init__(self):
        """Initialize the audio capture system."""
        print("🔧 Initializing audio processor...")
        
        self.sample_rate = Config.SAMPLE_RATE
        self.chunk_size = Config.CHUNK_SIZE
        self.buffer_duration = Config.BUFFER_SECONDS
        
        self.audio_queue = queue.Queue(maxsize=10)
        self.recording = False
        self.stream = None
        
        # Voice detector reference (will be set externally)
        self.voice_detector = None
        
        if SOUNDDEVICE_AVAILABLE:
            print("✅ Audio processor initialized")
        else:
            print("⚠️  SoundDevice not available - audio will be disabled")
    
    def set_voice_detector(self, voice_detector):
        """Set reference to voice detector."""
        self.voice_detector = voice_detector
        print("🔗 Audio capture connected to voice detector")
    
    def start_capture(self):
        """Start audio capture in separate thread."""
        if not SOUNDDEVICE_AVAILABLE:
            print("⚠️  Cannot start audio - SoundDevice not available")
            return
        
        try:
            self.recording = True
            
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            
            self.stream.start()
            print("✅ Audio capture started - speak to test voice detection!")
            
            # Processing loop - CRITICAL FOR REAL-TIME
            audio_buffer = []
            buffer_size = self.sample_rate * 3  # 3 seconds of audio
            
            while self.recording:
                try:
                    # Get audio chunk
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(audio_chunk)
                    
                    # Process when we have enough audio (3 seconds)
                    if len(audio_buffer) >= buffer_size:
                        # Convert to numpy array
                        audio_array = np.array(audio_buffer[-buffer_size:])  # Keep last 3 seconds
                        
                        # Process with voice detector
                        if self.voice_detector:
                            self.voice_detector.process_audio(audio_array)
                            print("🎤 Processing voice audio...")  # Debug
                        
                        # Keep only last 1 second for next iteration (sliding window)
                        audio_buffer = audio_buffer[-self.sample_rate:]
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    
        except Exception as e:
            print(f"❌ Audio capture error: {e}")
            self.recording = False
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio stream callback."""
        if status:
            print(f"Audio status: {status}")
        
        if self.recording:
            # Get mono audio
            audio_chunk = indata[:, 0]
            
            # Add to queue
            try:
                self.audio_queue.put(audio_chunk, block=False)
            except queue.Full:
                # Remove old data if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(audio_chunk, block=False)
                except queue.Empty:
                    pass
    
    def get_audio_chunk(self):
        """Get the latest audio chunk."""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop audio capture."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("🔇 Audio capture stopped")

