import torch
import torchaudio.functional as F
import numpy as np
import random
import librosa

def augment_audio(waveform, sample_rate):
    is_numpy = isinstance(waveform, np.ndarray)  
    
    if is_numpy:
        waveform = torch.FloatTensor(waveform)

    augmentations = [
        lambda x: change_volume(x),
        lambda x: add_noise(x),
        lambda x: change_tempo(x),
        lambda x: change_pitch(x, sample_rate),
    ]
    
    num_augmentations = random.randint(1, 3)
    selected_augmentations = random.sample(augmentations, num_augmentations)
    
    for aug_fn in selected_augmentations:
        waveform = aug_fn(waveform)

    if is_numpy:
        waveform = waveform.numpy()       

    return waveform

def change_volume(waveform):
    gain = torch.FloatTensor([random.uniform(0.7, 2)])
    return waveform * gain

def add_noise(waveform): 
    noise_level = random.uniform(0.001, 0.005)
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def change_tempo(waveform):    
    tempo_factor = random.uniform(0.85, 1.15)
    waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
    waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=tempo_factor)
    waveform_stretched = torch.from_numpy(waveform_stretched).float() if isinstance(waveform, torch.Tensor) else waveform_stretched
    
    return waveform_stretched

def change_pitch(waveform, sample_rate): 
    pitch_shift = random.randint(-2, 2) 

    waveform_shifted = F.pitch_shift(
        waveform.unsqueeze(0), 
        sample_rate, 
        pitch_shift
    ).squeeze(0)
    return waveform_shifted