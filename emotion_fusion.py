"""Emotion fusion engine combining face and voice predictions."""

from config import Config

class EmotionFusion:
    """Fuses face and voice emotions for best accuracy."""
    
    def __init__(self):
        self.face_weight = Config.FACE_WEIGHT
        self.voice_weight = Config.VOICE_WEIGHT
    
    def fuse_emotions(self, face_result, voice_result):
        """Combine face and voice emotions."""
        # Handle missing inputs
        if not face_result and not voice_result:
            return None
        
        if not face_result:
            return {
                'emotion': voice_result['emotion'],
                'confidence': voice_result['confidence'] * 0.8,
                'source': 'voice_only'
            }
        
        if not voice_result:
            return {
                'emotion': face_result['emotion'],
                'confidence': face_result['confidence'] * 0.8,
                'source': 'face_only'
            }
        
        # Both available - perform weighted fusion
        face_emotion = face_result['emotion']
        voice_emotion = voice_result['emotion']
        face_conf = face_result['confidence']
        voice_conf = voice_result['confidence']
        
        # Check if emotions agree
        if face_emotion == voice_emotion:
            # Boost confidence for agreement
            combined_confidence = min(1.0, 
                face_conf * self.face_weight + 
                voice_conf * self.voice_weight + 0.2
            )
            return {
                'emotion': face_emotion,
                'confidence': combined_confidence,
                'source': 'agreement'
            }
        else:
            # Use weighted selection
            face_weighted = face_conf * self.face_weight
            voice_weighted = voice_conf * self.voice_weight
            
            if face_weighted > voice_weighted:
                return {
                    'emotion': face_emotion,
                    'confidence': face_weighted,
                    'source': 'face_dominant'
                }
            else:
                return {
                    'emotion': voice_emotion,
                    'confidence': voice_weighted,
                    'source': 'voice_dominant'
                }

