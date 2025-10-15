
"""Simple, working audio capture without complex dependencies."""

import numpy as np
import threading
import queue
import time

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ SoundDevice loaded successfully")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  SoundDevice not available")

class AudioCapture:
    """Simple, reliable audio capture class."""
    
    def __init__(self):
        """Initialize audio capture."""
        print("🔧 Initializing simple audio capture...")
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.buffer_duration = 3
        
        # Audio processing
        self.audio_queue = queue.Queue(maxsize=10)
        self.recording = False
        self.stream = None
        
        # Voice detector reference
        self.voice_detector = None
        
        print("✅ Simple audio capture initialized")
    
    def set_voice_detector(self, voice_detector):
        """Connect voice detector."""
        self.voice_detector = voice_detector
        print("🔗 Audio capture connected to voice detector")
    
    def start_capture(self):
        """Start audio capture in separate thread."""
        if not SOUNDDEVICE_AVAILABLE:
            print("⚠️  Cannot start audio - SoundDevice not available")
            return
        
        print("🎤 Starting audio capture...")
        
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
            print("✅ Audio capture started - listening for voice...")
            
            # Simple processing loop
            audio_buffer = []
            target_samples = self.sample_rate * self.buffer_duration  # 3 seconds
            
            while self.recording:
                try:
                    # Get audio chunk from queue
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Add to buffer
                    audio_buffer.extend(audio_chunk)
                    
                    # Process when we have enough audio
                    if len(audio_buffer) >= target_samples:
                        # Convert to numpy array
                        audio_array = np.array(audio_buffer[-target_samples:])
                        
                        # Send to voice detector
                        if self.voice_detector:
                            self.voice_detector.process_audio(audio_array)
                        
                        # Keep sliding window (keep last 1 second)
                        audio_buffer = audio_buffer[-self.sample_rate:]
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    
        except Exception as e:
            print(f"❌ Audio capture error: {e}")
            self.recording = False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio status: {status}")
        
        if self.recording:
            # Get mono audio data
            audio_chunk = indata[:, 0]  # First channel only
            
            # Add to queue (non-blocking)
            try:
                self.audio_queue.put(audio_chunk, block=False)
            except queue.Full:
                # If queue is full, remove oldest and add new
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(audio_chunk, block=False)
                except queue.Empty:
                    pass
    
    def get_audio_chunk(self):
        """Get latest audio chunk (if needed)."""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop audio capture."""
        print("🔇 Stopping audio capture...")
        self.recording = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        print("✅ Audio capture stopped")

# Test the class
if __name__ == "__main__":
    print("Testing AudioCapture class...")
    
    # Test basic creation
    try:
        ac = AudioCapture()
        print("✅ AudioCapture created successfully!")
        
        # Test voice detector connection
        class DummyVoiceDetector:
            def process_audio(self, audio):
                print(f"Received audio: {len(audio)} samples")
        
        dummy_detector = DummyVoiceDetector()
        ac.set_voice_detector(dummy_detector)
        print("✅ Voice detector connection test passed!")
        
    except Exception as e:
        print(f"❌ AudioCapture test failed: {e}")
