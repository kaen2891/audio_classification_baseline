#from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
import torchaudio


class DepressedDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        
        print('train_flag', train_flag)
        
        if train_flag:
            annotation_file = os.path.join(args.data_folder, args.train_annotation_file)
        else:
            annotation_file = os.path.join(args.data_folder, args.test_annotation_file)
        
        
        self.train_flag = train_flag
        self.args = args

        # parameters for spectrograms
        self.sample_rate = args.sample_rate

        """ get dataset information """        
        annotation_file = pd.read_csv(annotation_file)        
        self.data_inputs = self.get_audio(annotation_file)
    
    def get_audio(self, annotation_file):
        df = pd.read_csv(annotation_file)
        files = df['연구번호'].values.tolist()
        labels = df['PHQ9_B'].values.tolist()
        
        data_inputs = list(zip(files, labels))
        return data_inputs

    def __getitem__(self, index):
        audio, labels = self.data_inputs[index]
        waveform, sample_rate = torchaudio.load(os.path.join(self.args.data_folder, audio))
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        input_features = self.processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=self.sample_rate).input_features.squeeze(0)

        return input_features, labels


    def __len__(self):
        return len(self.data_inputs)
