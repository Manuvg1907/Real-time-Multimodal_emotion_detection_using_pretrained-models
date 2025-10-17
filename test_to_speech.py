"""Text-to-speech that speaks EVERY emotion detected."""

import threading
import time
import queue

try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✅ TTS loaded successfully")
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  TTS not available")

class TextToSpeech:
    """TTS that speaks EVERY emotion without restrictions."""
    
    def __init__(self):
        self.enabled = True
        self.engine = None
        self.speak_queue = queue.Queue(maxsize=50)  # Larger queue
        self.speaking = False
        self.last_speak_time = 0
        self.emotion_count = 0
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 200)  # Fast speech
                self.engine.setProperty('volume', 1.0)  # Max volume
                
                # Get available voices
                voices = self.engine.getProperty('voices')
                if voices and len(voices) > 0:
                    self.engine.setProperty('voice', voices[0].id)  # Use first voice
                
                # Start multiple speaking workers for faster processing
                for i in range(2):  # 2 worker threads
                    worker = threading.Thread(target=self._speaking_worker, daemon=True)
                    worker.start()
                
                print("✅ ALWAYS-SPEAKING TTS engine initialized")
                
                # Test immediately
                self.speak_now("TTS ready")
                
            except Exception as e:
                print(f"⚠️  TTS initialization error: {e}")
                self.engine = None
    
    def speak_emotion(self, emotion, confidence):
        """Queue emotion for speaking (OLD METHOD - still works)."""
        self.speak_emotion_now(emotion, confidence)
    
    def speak_emotion_now(self, emotion, confidence):
        """Speak emotion IMMEDIATELY without any restrictions."""
        if not self.enabled or not self.engine:
            return
        
        self.emotion_count += 1
        current_time = time.time()
        
        # Create short, clear message
        messages = [
            f"{emotion}",  # Just the emotion name
            f"{emotion} detected",
            f"Emotion {emotion}",
            f"{emotion} emotion"
        ]
        
        # Rotate through different message formats
        message = messages[self.emotion_count % len(messages)]
        
        # Add to queue immediately
        try:
            self.speak_queue.put((message, current_time), block=False)
            print(f"🔊 QUEUED: {message} at {current_time:.1f}")
        except queue.Full:
            # If queue is full, clear old items and add new one
            try:
                while self.speak_queue.qsize() > 10:
                    self.speak_queue.get_nowait()
                self.speak_queue.put((message, current_time), block=False)
            except:
                pass
        
        self.last_speak_time = current_time
    
    def _speaking_worker(self):
        """Background worker that speaks ALL queued messages."""
        worker_id = threading.current_thread().ident
        print(f"🔊 TTS Worker {worker_id} started")
        
        while True:
            try:
                # Get message from queue (blocking with timeout)
                message_data = self.speak_queue.get(timeout=2.0)
                message, queue_time = message_data
                
                if self.engine and self.enabled:
                    current_time = time.time()
                    print(f"🔊 SPEAKING NOW: {message} (queued {current_time - queue_time:.1f}s ago)")
                    
                    # Speak immediately
                    self.speaking = True
                    try:
                        self.engine.say(message)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"Speech error: {e}")
                    finally:
                        self.speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}")
                self.speaking = False
    
    def speak_now(self, message):
        """Speak a message RIGHT NOW (bypass queue)."""
        if self.engine and self.enabled:
            try:
                print(f"🔊 IMMEDIATE: {message}")
                
                # Use a separate engine instance for immediate speech
                immediate_engine = pyttsx3.init()
                immediate_engine.setProperty('rate', 220)
                immediate_engine.setProperty('volume', 1.0)
                immediate_engine.say(message)
                immediate_engine.runAndWait()
                immediate_engine.stop()
                
            except Exception as e:
                print(f"Immediate TTS error: {e}")
    
    def speak_continuous(self, emotion):
        """Speak emotion continuously (for testing)."""
        for i in range(5):
            self.speak_emotion_now(emotion, 0.8)
            time.sleep(1)
    
    def toggle_speech(self):
        """Toggle speech on/off."""
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f"🔊 TTS: {status}")
        
        # Immediate feedback
        self.speak_now(f"TTS {status}")
    
    def test_continuous_speech(self):
        """Test continuous speech."""
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'neutral', 'disgust']
        print("🔊 Testing continuous speech...")
        
        for emotion in emotions:
            self.speak_emotion_now(emotion, 0.8)
            time.sleep(1.5)  # Short delay between emotions
    
    def clear_queue(self):
        """Clear the speaking queue."""
        while not self.speak_queue.empty():
            try:
                self.speak_queue.get_nowait()
            except queue.Empty:
                break
        print("🔊 Speech queue cleared")
    
    def stop(self):
        """Stop TTS engine."""
        self.enabled = False
        self.clear_queue()
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

# Test continuous TTS when run directly
if __name__ == "__main__":
    print("Testing CONTINUOUS TTS...")
    tts = TextToSpeech()
    
    if tts.engine:
        print("Testing continuous emotion speech...")
        tts.test_continuous_speech()
        print("Continuous TTS test complete!")
    else:
        print("TTS not available for testing")

