# -*- coding: cp949 -*-
#from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from glob import glob

from collections import Counter
from sklearn.utils import resample

import torch
from torch.utils.data import Dataset
import torchaudio
torchaudio.set_audio_backend("soundfile") 

class DepressedDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        
        print('train_flag', train_flag)
        
        self.processor = args.processor
        
        if train_flag:
            annotation_file = os.path.join(args.data_folder, args.train_annotation_file)
        else:
            annotation_file = os.path.join(args.data_folder, args.test_annotation_file)
        print('annotation', annotation_file)
        
        self.train_flag = train_flag
        self.args = args
        self.class_nums = np.zeros(self.args.n_cls)
        self.gender_nums = np.zeros(2) # male vs. female
        self.age_nums = np.zeros(2) # adult vs. elderly

        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate

        """ get dataset information """
        self.data_inputs = self.get_audio(annotation_file)
        
    def get_audio(self, annotation_file):
        df = pd.read_csv(annotation_file)
        files = df['연구번호'].values.tolist()
        labels = df['PHQ9_B'].values.tolist()
        genders = df['Sex'].values.tolist()
        ages = df['Age'].values.tolist()
          
        
        for label in labels:
            if int(label) >= 1:
                label = 1
            else:
                label = 0
            self.class_nums[label] += 1
                
        for gender in genders:
            if int(gender) == 1:
                gender = 0
            else:
                gender = 1
            self.gender_nums[gender] += 1
        
        for age in ages:
            if int(age) < 65:
                age = 0
            else:
                age = 1
            self.age_nums[age] += 1
        
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        print('class_ratio is', self.class_ratio)
        
        self.gender_ratio = self.gender_nums / sum(self.gender_nums) * 100
        print('gender_ratio is', self.gender_ratio)
        
        self.age_ratio = self.age_nums / sum(self.age_nums) * 100
        print('age_ratio is', self.age_ratio)
        
        #data_inputs = list(zip(files, labels))
        data_inputs = list(zip(files, labels, genders, ages))
        return data_inputs

    def __getitem__(self, index):
        #audio, labels = self.data_inputs[index]
        audio, label, gender, age = self.data_inputs[index]
        '''
        if int(label) >= 1:
            label = 1
        '''
        
        if int(gender) == 1:
            gender = 0
        else:
            gender = 1
        
        if int(age) < 65:
            age = 0
        else:
            age = 1
        
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)
        age = torch.tensor(age, dtype=torch.long)
        
        waveform, sample_rate = torchaudio.load(os.path.join(self.args.data_folder, audio))
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        
        if waveform.size(0) >= 2: # the speech is stereo channel data, thus make this sample as mono channel 
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if waveform.size(1) >= 2880000: # if the data length is longer than 180 secs, we cut the sample until 180 secs
            waveform = waveform[:, :2880000] 
        
        input_features = self.processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=self.sample_rate).input_values.squeeze(0)

        return input_features, label, gender, age


    def __len__(self):
        return len(self.data_inputs)
    


class AugDepressedDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        
        print('train_flag', train_flag)
        
        self.processor = args.processor
        
        if train_flag:
            annotation_file = os.path.join(args.data_folder, args.train_annotation_file)
        else:
            annotation_file = os.path.join(args.data_folder, args.test_annotation_file)
        print('annotation', annotation_file)
        
        self.train_flag = train_flag
        self.args = args
        self.class_nums = np.zeros(self.args.n_cls)
        self.gender_nums = np.zeros(2) # male vs. female
        self.age_nums = np.zeros(2) # adult vs. elderly
        
        self.class_nums_over = np.zeros(self.args.n_cls)
        self.gender_nums_over = np.zeros(2) # male vs. female
        self.age_nums_over = np.zeros(2) # adult vs. elderly

        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate

        """ get dataset information """
        self.data_inputs = self.get_audio(annotation_file)
    
    def apply_augmentation(self, waveform, sample_rate):
        # random augmentation
        if random.random() < 0.5:
            waveform = self.add_noise(waveform)
        if random.random() < 0.3:
            waveform = self.time_stretch(waveform)
        '''
        if random.random() < 0.3:
            waveform = self.pitch_shift(waveform, sample_rate)
        '''
        return waveform

    def add_noise(self, waveform, noise_level=0.005):
        noise = noise_level * torch.randn_like(waveform)
        return waveform + noise

    def time_stretch(self, waveform, rate_range=(0.9, 1.1)):
        rate = random.uniform(*rate_range)
        stretched = torchaudio.functional.phase_vocoder(waveform.unsqueeze(0), rate=rate, phase_advance=torch.zeros(1))
        return stretched.squeeze(0)

    def pitch_shift(self, waveform, sample_rate, n_steps_range=(-2, 2)):
        n_steps = random.uniform(*n_steps_range)
        effect = [['pitch', str(n_steps * 100)], ['rate', str(sample_rate)]]
        waveform_shifted, _ = torchaudio.sox_effects.apply_effects_tensor(waveform.unsqueeze(0), sample_rate, effect)
        return waveform_shifted.squeeze(0)
        
    def get_audio(self, annotation_file):
        df = pd.read_csv(annotation_file)
        files = df['연구번호'].values.tolist()
        labels = df['PHQ9_B'].values.tolist()
        genders = df['Sex'].values.tolist()
        ages = df['Age'].values.tolist()
        
        new_labels = []
        new_genders = []
        new_ages = []
        
        for label in labels:
            if int(label) >= 1:
                label = 1
            else:
                label = 0
            self.class_nums[label] += 1
            new_labels.append(label)
        
        for gender in genders:
            if int(gender) == 1:
                gender = 0
            else:
                gender = 1
            self.gender_nums[gender] += 1
            new_genders.append(gender)
        
        for age in ages:
            if int(age) < 65:
                age = 0
            else:
                age = 1
            self.age_nums[age] += 1
            new_ages.append(age)
                
        
        suicidal_class = 1
        suicidal_audio_paths = [p for p, l in zip(files, new_labels) if l == suicidal_class]
        suicidal_labels = [1 for _ in range(len(suicidal_audio_paths))]
        suicidal_genders = [g for g, l in zip(new_genders, new_labels) if l == suicidal_class]
        suicidal_ages = [a for a, l in zip(new_ages, new_labels) if l == suicidal_class]
        
        hc_class_count = Counter(new_labels)[0]
        
        
        oversampled_audio_paths, oversampled_labels, oversampled_genders, oversampled_ages = resample(
            suicidal_audio_paths,
            suicidal_labels,
            suicidal_genders,
            suicidal_ages,
            replace=True,
            n_samples=hc_class_count,
            random_state=self.args.seed
        )
        
        final_files = files + oversampled_audio_paths
        final_labels = new_labels + oversampled_labels
        final_genders = new_genders + oversampled_genders
        final_ages = new_ages + oversampled_ages
        
        
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        #print('class_ratio is', self.class_ratio)
        
        self.gender_ratio = self.gender_nums / sum(self.gender_nums) * 100
        #print('gender_ratio is', self.gender_ratio)
        
        self.age_ratio = self.age_nums / sum(self.age_nums) * 100
        #print('age_ratio is', self.age_ratio)
        
        
        new_labels_over = []
        new_genders_over = []
        new_ages_over = []
        
        #print(oversampled_ages)
        
        for label in final_labels:
            if int(label) >= 1:
                label = 1
            else:
                label = 0
            self.class_nums_over[label] += 1
            new_labels_over.append(label)
        
        for gender in final_genders:
            self.gender_nums_over[gender] += 1
            new_genders_over.append(gender)
        
        for age in final_ages:
            self.age_nums_over[age] += 1
            new_ages_over.append(age)
        
        self.class_ratio_over = self.class_nums_over / sum(self.class_nums_over) * 100
        print('class_ratio_over is', self.class_ratio_over)
        
        self.gender_ratio_over = self.gender_nums_over / sum(self.gender_nums_over) * 100
        print('gender_ratio_over is', self.gender_ratio_over)
        
        self.age_ratio_over = self.age_nums_over / sum(self.age_nums_over) * 100
        print('age_ratio_over is', self.age_ratio_over)
        
        
        #data_inputs = list(zip(files, labels))
        data_inputs = list(zip(final_files, new_labels_over, new_genders_over, new_ages_over))
        return data_inputs

    def __getitem__(self, index):
        #audio, labels = self.data_inputs[index]
        audio, label, gender, age = self.data_inputs[index]
        
        label = torch.tensor(label, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)
        age = torch.tensor(age, dtype=torch.long)
        
        waveform, sample_rate = torchaudio.load(os.path.join(self.args.data_folder, audio))
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        
        if waveform.size(0) >= 2: # the speech is stereo channel data, thus make this sample as mono channel 
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if waveform.size(1) >= 2880000: # if the data length is longer than 180 secs, we cut the sample until 180 secs
            waveform = waveform[:, :2880000]
        
        waveform = self.apply_augmentation(waveform, sample_rate)
        
        
        input_features = self.processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=self.sample_rate).input_values.squeeze(0)

        return input_features, label, gender, age


    def __len__(self):
        return len(self.data_inputs)
