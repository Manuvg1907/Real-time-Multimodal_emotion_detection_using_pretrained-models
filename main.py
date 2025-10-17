"""
Real-time Multimodal Emotion Detection System
DeepFace (99.6%) + Wav2Vec2-XLSR (75%+) Fusion
UPDATED for Continuous Voice Output
"""

import cv2
import threading
import time
import sys
from pathlib import Path

# Import our modules
from config import Config
from face_detector import FaceDetector
from voice_detector import VoiceDetector  
from emotion_fusion import EmotionFusion
from audio_capture import AudioCapture
from text_to_speech import TextToSpeech

class MultimodalEmotionDetector:
    """Main application class with continuous voice feedback."""
    
    def __init__(self):
        print("🎭 Initializing Multimodal Emotion Detection System...")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.voice_detector = VoiceDetector()
        self.emotion_fusion = EmotionFusion()
        self.audio_capture = AudioCapture()
        self.tts = TextToSpeech()
        
        # CRITICAL: Connect audio capture to voice detector
        self.audio_capture.set_voice_detector(self.voice_detector)
        
        # State variables for continuous speaking
        self.running = False
        self.current_emotion = None
        self.last_spoken_time = 0
        self.last_face_emotion = None
        self.last_voice_emotion = None
        self.emotion_speak_count = 0
        
        print("✅ System initialized successfully!")
        print(f"📊 Expected Accuracy: Face={Config.FACE_ACCURACY}, Voice={Config.VOICE_ACCURACY}")
        
        # Test TTS immediately on startup
        self.tts.speak_now("Emotion detection system ready")
    
    def start(self):
        """Start the real-time detection system with continuous voice output."""
        print("\n🚀 Starting real-time emotion detection...")
        print("📹 Opening camera...")
        print("🎤 Starting audio capture...")
        print("👁️  Press 'q' to quit, 's' to toggle speech, 't' to test TTS")
        print("-" * 60)
        
        self.running = True
        
        # CRITICAL: Start audio capture in separate thread
        audio_thread = threading.Thread(target=self.audio_capture.start_capture, daemon=True)
        audio_thread.start()
        print("🎤 Audio thread started")
        
        # Give audio thread time to initialize
        time.sleep(1)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return
        
        try:
            frame_count = 0
            last_debug_time = time.time()
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                current_time = time.time()
                
                # Detect face emotion
                face_result = self.face_detector.detect_emotion(frame)
                
                # Get REAL-TIME voice emotion
                voice_result = self.voice_detector.get_current_emotion()
                
                # Fuse emotions EVERY frame
                final_emotion = self.emotion_fusion.fuse_emotions(face_result, voice_result)
                
                # CONTINUOUS emotion speaking - check every frame
                self.continuous_speak_emotion(final_emotion, face_result, voice_result, current_time)
                
                # Display results on frame
                frame = self.draw_results(frame, face_result, voice_result, final_emotion)
                
                # DEBUG: Print what we're getting every 3 seconds
                if current_time - last_debug_time > 3.0:
                    print("\n" + "="*50)
                    if face_result:
                        print(f"👁️  Face: {face_result['emotion']} ({face_result['confidence']:.2f})")
                    if voice_result:
                        print(f"🎤 Voice: {voice_result['emotion']} ({voice_result['confidence']:.2f})")
                    else:
                        print("🔇 No voice input detected")
                    if final_emotion:
                        print(f"🎭 FINAL: {final_emotion['emotion']} ({final_emotion['confidence']:.2f})")
                    print(f"🔊 Speech Count: {self.emotion_speak_count}")
                    print("="*50)
                    last_debug_time = current_time
                
                # Show frame
                cv2.imshow('Multimodal Emotion Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.tts.toggle_speech()
                elif key == ord('t'):  # Test TTS
                    self.test_tts_functionality()
                elif key == ord('c'):  # Clear and test continuous speech
                    self.tts.test_continuous_speech()
                
                frame_count += 1
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.cleanup(cap)
    
    def continuous_speak_emotion(self, final_emotion, face_result, voice_result, current_time):
        """Continuously speak emotions with multiple triggers."""
        if not self.tts.enabled:
            return
        
        # Multiple speaking triggers
        should_speak = False
        speak_reason = ""
        
        # 1. Final emotion changed
        if final_emotion and (self.current_emotion != final_emotion['emotion']):
            if final_emotion['confidence'] > 0.4:
                should_speak = True
                speak_reason = "emotion_changed"
        
        # 2. Face emotion changed (speak face-specific)
        if face_result and (self.last_face_emotion != face_result['emotion']):
            if face_result['confidence'] > 0.7:
                should_speak = True
                speak_reason = "face_changed"
                self.last_face_emotion = face_result['emotion']
        
        # 3. Voice emotion changed (speak voice-specific)  
        if voice_result and (self.last_voice_emotion != voice_result['emotion']):
            if voice_result['confidence'] > 0.5:
                should_speak = True
                speak_reason = "voice_changed"
                self.last_voice_emotion = voice_result['emotion']
        
        # 4. Time-based speaking (every 2 seconds regardless)
        time_passed = (current_time - self.last_spoken_time) > 2.0
        if time_passed and final_emotion and final_emotion['confidence'] > 0.4:
            should_speak = True
            speak_reason = "time_based"
        
        # 5. High confidence trigger (speak immediately for high confidence)
        if final_emotion and final_emotion['confidence'] > 0.85:
            time_since_last = current_time - self.last_spoken_time
            if time_since_last > 0.5:  # At least 0.5 seconds between high confidence
                should_speak = True
                speak_reason = "high_confidence"
        
        # Execute speaking
        if should_speak and final_emotion:
            self.execute_emotion_speaking(final_emotion, speak_reason, current_time)
    
    def execute_emotion_speaking(self, final_emotion, reason, current_time):
        """Execute the actual speaking with variety."""
        try:
            self.emotion_speak_count += 1
            
            # Create varied messages
            emotion = final_emotion['emotion']
            confidence = final_emotion['confidence']
            
            # Different message formats for variety
            messages = [
                f"{emotion}",
                f"{emotion} detected",
                f"Emotion {emotion}",
                f"{emotion} emotion",
                f"Current emotion {emotion}",
                f"Feeling {emotion}",
                f"I see {emotion}",
                f"{emotion} expression"
            ]
            
            # Select message based on reason and count
            if reason == "face_changed":
                message = f"Face shows {emotion}"
            elif reason == "voice_changed":
                message = f"Voice shows {emotion}" 
            elif reason == "high_confidence":
                message = f"Strong {emotion}"
            else:
                message = messages[self.emotion_speak_count % len(messages)]
            
            # Speak with high priority
            self.tts.speak_emotion_now(emotion, confidence)
            
            # Update state
            self.current_emotion = emotion
            self.last_spoken_time = current_time
            
            print(f"🔊 SPOKE[{self.emotion_speak_count}]: {message} ({reason}) at {current_time:.1f}")
            
        except Exception as e:
            print(f"Speaking error: {e}")
    
    def test_tts_functionality(self):
        """Test TTS with all emotions."""
        print("🔊 Testing TTS functionality...")
        self.tts.speak_now("Testing TTS now")
        
        # Test all emotions quickly
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'neutral', 'disgust']
        for emotion in emotions:
            self.tts.speak_emotion_now(emotion, 0.8)
            time.sleep(0.8)  # Short delay
    
    def draw_results(self, frame, face_result, voice_result, final_emotion):
        """Draw emotion results on frame with enhanced info."""
        height, width = frame.shape[:2]
        
        # Draw face detection box if available
        if face_result and face_result.get('face_box'):
            x, y, w, h = face_result['face_box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw face emotion
            face_text = f"Face: {face_result['emotion']} ({face_result['confidence']:.2f})"
            cv2.putText(frame, face_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw voice emotion (top-left) - REAL-TIME STATUS
        if voice_result:
            voice_text = f"Voice: {voice_result['emotion']} ({voice_result['confidence']:.2f})"
            cv2.putText(frame, voice_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Show that we're listening but no voice detected
            cv2.putText(frame, "Voice: Listening...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Draw final emotion (center-top, large)
        if final_emotion:
            final_text = f"EMOTION: {final_emotion['emotion'].upper()}"
            confidence_text = f"Confidence: {final_emotion['confidence']:.2f}"
            source_text = f"Source: {final_emotion.get('source', 'fused')}"
            
            # Calculate text size for centering
            (text_width, text_height), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x = (width - text_width) // 2
            
            # Draw background rectangle
            cv2.rectangle(frame, (text_x-10, 60), (text_x+text_width+10, 140), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, final_text, (text_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, confidence_text, (text_x, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, source_text, (text_x, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw enhanced status info (bottom-left)
        status_y = height - 60
        cv2.putText(frame, f"TTS: {'ON' if self.tts.enabled else 'OFF'}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw audio status
        audio_status = "🎤 ACTIVE" if voice_result else "🎤 Listening"
        cv2.putText(frame, audio_status, (10, status_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if voice_result else (128, 128, 128), 2)
        
        # Draw speech counter
        cv2.putText(frame, f"Spoke: {self.emotion_speak_count}", (10, status_y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw controls
        cv2.putText(frame, "q=quit s=toggle_tts t=test c=continuous", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def maybe_speak_emotion(self, final_emotion):
        """DEPRECATED: Replaced by continuous_speak_emotion."""
        # This method is now handled by continuous_speak_emotion
        pass
    
    def cleanup(self, cap):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.audio_capture.stop_capture()
        self.tts.stop()
        print("✅ Cleanup complete")
        print(f"📊 Total emotions spoken: {self.emotion_speak_count}")

def main():
    """Main function."""
    try:
        detector = MultimodalEmotionDetector()
        detector.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

