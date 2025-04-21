from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T

__all__ = ['save_image', 'get_mean_and_std', 'get_individual_samples_torchaudio', 'get_individual_samples_torchaudio_multi_channel', 'get_individual_samples_torchaudio_batch', 'generate_fbank', 'get_score']


# ==========================================================================

def get_mean_and_std(dataset):
    """ Compute the mean and std value of mel-spectrogram """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    cnt = 0
    fst_moment = torch.zeros(1)
    snd_moment = torch.zeros(1)
    for inputs, _, _ in dataloader:
        b, c, h, w = inputs.shape
        nb_pixels = b * h * w

        fst_moment += torch.sum(inputs, dim=[0,2,3])
        snd_moment += torch.sum(inputs**2, dim=[0,2,3])
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment/cnt - mean**2)

    return mean, std
# ==========================================================================


# ==========================================================================
""" data preprocessing """

def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data


def get_individual_samples_torchaudio(args, sample, sample_rate, label):
    
    data, sr = torchaudio.load(sample)
    
    if data.size(0) == 2: # if stereo 
        data = torch.mean(data, dim=0).unsqueeze(0)
    
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)
        
    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    data = cut_pad_sample_torchaudio(data, args)
    
    return data, label
    

def get_individual_samples_torchaudio_multi_channel(args, sample_list, sample_rate, label):
    
    data_list = []
    for data in sample_list:
        data, sr = torchaudio.load(data)
        
        if data.size(0) == 2: # if stereo 
            data = torch.mean(data, dim=0).unsqueeze(0)
        
        if sr != sample_rate:
            resample = T.Resample(sr, sample_rate)
            data = resample(data)
            
        fade_samples_ratio = 16
        fade_samples = int(sample_rate / fade_samples_ratio)
        fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
        data = fade(data)
        data = cut_pad_sample_torchaudio(data, args)
        data_list.append(data)
    
    audio_tensor = torch.stack(data_list)
    audio_tensor = audio_tensor.squeeze(1)  # (num_channels, samples)
    output_audio = audio_tensor.mean(dim=0, keepdim=True)
    
    return output_audio, label


def get_individual_samples_torchaudio_batch(args, samples, sample_rate, label):
    
    sample_data = []
    
    samples = samples.split(',')
    #print(samples)
    
    for sample in samples:
        sample = sample.replace(" ", "")
        #print(sample)
        data, sr = torchaudio.load(sample)
        
        
        if data.size(0) == 2: # if stereo 
            data = torch.mean(data, dim=0).unsqueeze(0)
        
        if sr != sample_rate:
            resample = T.Resample(sr, sample_rate)
            data = resample(data)
            
        fade_samples_ratio = 16
        fade_samples = int(sample_rate / fade_samples_ratio)
        fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
        data = fade(data)
        data = cut_pad_sample_torchaudio(data, args)
        
        sample_data.append(data)
    
    return sample_data, label


def generate_fbank(args, audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    if args.model in ['ast']:
        mean, std =  -4.2677393, 4.5689974
    else:
        mean, std = fbank.mean(), fbank.std()
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    print(hits)
    print(counts)
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
