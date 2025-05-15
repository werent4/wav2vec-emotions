import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from dataset import EMOTION2ID

class Wav2VecEmotionClassifier:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")#model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(EMOTION2ID),
            id2label={i: label for label, i in EMOTION2ID.items()},
            label2id=EMOTION2ID
        ).to(self.device)

    def predict_emotion(self, audio, target_sr, duration_seconds=5):
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=target_sr, 
            return_tensors="pt", 
            padding="max_length",
            max_length=target_sr * duration_seconds,
            truncation=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        emotion_id = predictions.item()
        emotion_name = self.model.config.id2label[emotion_id]
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        return emotion_name, {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}

def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = Wav2VecEmotionClassifier(model_name=args.model_name, device=device)

    audio = load_audio(args.audio_file_path)
    emotion, probabilities = classifier.predict_emotion(audio, args.target_sr, args.duration_seconds)
    
    print(f"Emotion: {emotion}")
    print("\nProbs:")
    for emotion_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion_name}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotion Classification using Wav2Vec2")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--audio_file_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sampling rate for audio")
    parser.add_argument("--duration_seconds", type=int, default=5, help="Duration in seconds for audio input")

    args = parser.parse_args()
    main(args)
