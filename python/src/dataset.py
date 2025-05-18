import torch, torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Any
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
import warnings
from tqdm import tqdm
import random

from augments import augment_audio

EMOTION2ID = {
    'angry': 0,
    'sad': 1,
    'neutral': 2,
    'happy': 3,
    'excited': 4,  
    'frustrated': 5,
    'fear': 6,
    'surprise': 7,
    'disgust': 8,
    'unknown': 9
}

SIMPLIFIED_EMOTION2ID = {
    'angry': 0,       # angry, frustrated, disgust -> negative active
    'sad': 1,         # sad, fear -> negative passive
    'neutral': 2,     # neutral
    'happy': 3,       # happy, excited, surprise -> positive 
    'unknown': 4
}

class IEMOCAPDataset(Dataset):
    def __init__(
            self,
            dataset: DatasetDict,
            emotion_mapping: Dict[str, int],
            feature_extractor: Wav2Vec2FeatureExtractor,
            is_simplified: bool = False,
            max_length: int = 16000 * 5, # 5 seconds
            use_augmentations: bool = True,
            max_samples_per_class: int = 1000,
        ) -> None:

        if is_simplified:
            self.emotion_to_simplified = {
                'angry': 'angry',       # negative active
                'frustrated': 'angry',  # negative active
                'disgust': 'angry',     # negative active
                
                'sad': 'sad',           # negative passive
                'fear': 'sad',          # negative passive
                
                'neutral': 'neutral',   
                
                'happy': 'happy',       # positive
                'excited': 'happy',     # positive
                'surprise': 'happy',    # positive
                
                'unknown': 'unknown'    
            }
        
        self.dataset = dataset
        self.emotion_mapping = emotion_mapping
        self.max_length = max_length
        self.use_augmentations = use_augmentations
        self.max_samples_per_class = max_samples_per_class

        self.feature_extractor = feature_extractor

        self._prepare_data(is_simplified)

    def _get_emotion_id(self, label: str, is_simplified: bool) -> int:
        if is_simplified:
            if label in self.emotion_to_simplified:
                simplified_label = self.emotion_to_simplified[label]
                return self.emotion_mapping[simplified_label]
            else:
                warnings.warn(f"Label {label} not found in simplified emotion mapping. Using 'unknown' as default.")
                return self.emotion_mapping['unknown']
        else:
            if label.lower() in self.emotion_mapping:
                return self.emotion_mapping[label]
            else:
                warnings.warn(f"Label {label} not found in emotion mapping. Using 'unknown' as default.")
                return self.emotion_mapping['unknown']
        
    def _prepare_data(self, is_simplified):
        self.emotions_dist = {}
        if self.max_length > 0:
            self.processed_audio = []
            
            for i in tqdm(range(len(self.dataset)), desc= "Preparing dataset", unit="file"):
                item = self.dataset[i]
                audio = item['audio']['array']
                sample_rate = item['audio']['sampling_rate']
                
                label = self._get_emotion_id(item['major_emotion'], is_simplified)
                if self.emotions_dist.get(label, 0) > self.max_samples_per_class:
                    continue
                self.emotions_dist[label] = self.emotions_dist.get(label, 0) + 1

                if self.use_augmentations and random.random() > 0.5:
                    audio = augment_audio(audio, sample_rate)

                inputs = self.feature_extractor(
                        audio, sampling_rate=self.feature_extractor.sampling_rate, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt", type=torch.float32
                    )["input_values"].squeeze(0)
                
                self.processed_audio.append({
                    'input_values': inputs,
                    'sampling_rate': sample_rate,
                    'label': label
                })
        
    def __len__(self):
        return len(self.processed_audio)

    def __getitem__(self, idx):
        item = self.processed_audio[idx]
        
        return {
            'input_values': item['input_values'],
            'label': item['label']
        }
    
def print_emotion_distribution(iemocap_dataset: IEMOCAPDataset, is_simplified: bool = False):
    original_emotions = {}
    for item in iemocap_dataset.dataset:
        emotion = item['major_emotion'].lower()
        if emotion not in original_emotions:
            original_emotions[emotion] = 0
        original_emotions[emotion] += 1

    processed_emotions = {}
    for item in iemocap_dataset.processed_audio:
        label_id = item['label']

        emotion_name = None
        for name, idx in iemocap_dataset.emotion_mapping.items():
            if idx == label_id:
                emotion_name = name
                break
        
        if emotion_name not in processed_emotions:
            processed_emotions[emotion_name] = 0
        processed_emotions[emotion_name] += 1
    
    total = len(iemocap_dataset.dataset)
    print(f"\n{'=' * 50}")
    print(f"Emmotion distribution in dataset (total: {total} emotions):")
    print(f"{'=' * 50}")
    
    print("\nOriginal distributution:")
    print(f"{'Emotion':<15} {'Count':<10} {'Procent':<10}")
    print('-' * 35)
    for emotion, count in sorted(original_emotions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{emotion:<15} {count:<10} {percentage:.2f}%")
        
    if is_simplified:
        print("\nAfter is_simplified:")
        print(f"{'Emotion':<15} {'Count':<10} {'Procent':<10}")
        print('-' * 35)
        for emotion, count in sorted(processed_emotions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"{emotion:<15} {count:<10} {percentage:.2f}%")
            
    print(f"\n{'=' * 50}")