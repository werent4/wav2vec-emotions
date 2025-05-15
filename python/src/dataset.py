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
}

class IEMOCAPDataset(Dataset):
    def __init__(
            self,
            dataset: DatasetDict,
            emotion_mapping: Dict[str, int],
            feature_extractor: Wav2Vec2FeatureExtractor,
            is_simplified: bool = False,
            max_length: int = 16000 * 5, # 5 seconds
            use_augmentations: bool = True
        ) -> None:

        if is_simplified:
            raise NotImplementedError("Only regular mapping is implemented.")
        
        self.dataset = dataset
        self.emotion_mapping = emotion_mapping
        self.max_length = max_length
        self.use_augmentations = use_augmentations

        self.feature_extractor = feature_extractor

        self._prepare_data()

    def _get_emotion_id(self, label: str) -> int:
        if label.lower() in self.emotion_mapping:
            return self.emotion_mapping[label]
        else:
            warnings.warn(f"Label {label} not found in emotion mapping. Using 'unknown' as default.")
            return self.emotion_mapping['unknown']
    
        
    def _prepare_data(self):
        if self.max_length > 0:
            self.processed_audio = []
            
            for i in tqdm(range(len(self.dataset)), desc= "Preparing dataset", unit="file"):
                item = self.dataset[i]
                audio = item['audio']['array']
                sample_rate = item['audio']['sampling_rate']
                label = self._get_emotion_id(item['major_emotion'])

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
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.processed_audio[idx]
        
        return {
            'input_values': item['input_values'],
            'label': item['label']
        }