import torch, torchaudio
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Any
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import warnings
from tqdm import tqdm
import random
import json

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

EMOTION2ID_GLICLASS = {
    'angry': 0,
    'sad': 1,
    'neutral': 2,
    'happy': 3,
    'disgusted': 4,
    'fearful': 5,
    'suprised': 6,
}

class GLiClassEmotionAudioDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            feature_extractor: Wav2Vec2FeatureExtractor,
            emotion_mapping: Dict[str, int] = EMOTION2ID_GLICLASS,
            sampling_rate: int = 16000,
            max_length_s: int = 5, # 5 seconds
            
        ) -> None:
        
        self.dataset = self._load_dataset(dataset_path)
        self.emotion_mapping = emotion_mapping
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate
        self.max_duration_samples = sampling_rate * max_length_s

        self._prepare_data()
    
    def _load_dataset(self, dataset_path):
        with open(dataset_path, "r", encoding= "utf-8") as f:
            data = json.load(f)
        return data

    def _prepare_data(self):
        new_rows = []
        for row in self.dataset:
            all_labels = [EMOTION2ID_GLICLASS[label.lower()] for label in row["all_labels"]] 
            label = [EMOTION2ID_GLICLASS[label.lower()] for label in row["true_labels"]] 
            if len(label) > 1:
                raise ValueError("ONLY ONE LABEL")
            new_row = {
                "id": row["id"], 
                "audio_path": row["audio_path"], 
                "sample_rate": row["sample_rate"], 
                "all_labels": all_labels, 
                "true_labels": label[0]
            }
            new_rows.append(new_row)
        self.dataset = new_rows

    def prepare_audio(self, audio_array, audio_sr):
        if isinstance(audio_array, np.ndarray):
            audio_array = torch.from_numpy(audio_array).float()
        elif isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.float()
        else:
            audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # if random.random() > 0.3:
        #     audio_array = augment_audio(audio_array, audio_sr)

        if audio_sr != self.sampling_rate:
            audio_array = Resample(audio_sr, new_freq= self.sampling_rate)(audio_array)

        audio_inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            # padding="longest",
            truncation=True, 
            max_length=self.max_duration_samples  
        )
        return audio_inputs["input_values"].squeeze(0) 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_array = torch.load(item['audio_path'])
        audio_inputs = self.prepare_audio(audio_array, item['sample_rate'])
        return {
            'input_values': audio_inputs,
            'labels': item['true_labels']
        }
    
@dataclass
class DataCollatorForWav2Vec2Classification:
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Union[int, None] = None
    pad_to_multiple_of: Union[int, None] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["labels"] for feature in features]
        
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        # print(batch)
        # import os; os._exit(-1)
        return batch


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