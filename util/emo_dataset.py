from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy

from .emo_util import generate_fbank, get_individual_samples_torchaudio, get_individual_samples_torchaudio_multi_channel, get_individual_samples_torchaudio_batch, cut_pad_sample_torchaudio

class EmoDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        
        print('train_flag', train_flag)
        
        if train_flag:
            annotation_file = os.path.join(args.data_folder, args.train_annotation_file)
        else:
            annotation_file = os.path.join(args.data_folder, args.test_annotation_file)
        
        
        self.train_flag = train_flag
        self.args = args
        self.ch = self.args.channel

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels

        """ get dataset information """        
        df = pd.read_csv(annotation_file)        
        files = df['file_path'].values.tolist()
        labels = df['label'].values.tolist()
        
        self.filenames = []
        self.audio_data = []  # each sample will be a tuple with (audio_data, label)

        if print_flag:
            print('*' * 20)  
            print("Extracting individual emotional data samples..")
        
        self.class_nums = np.zeros(args.n_cls)
        
        '''
        ch1: ch1
        ch2: ch1, ch4
        ch3: ch1, ch4, ch5
        ch4: ch1, ch3, ch4, ch5
        ch5: ch1, ch3, ch4, ch5, ch6
        ch6: ch1, ch3, ch4, ch5, ch6, ch7        
        '''
        
        
        for i, (data, label) in enumerate(zip(files, labels)):
            if self.ch > 0:
                wav_sample = os.path.join(args.data_folder, data)                                
                _dir, file_id = os.path.split(wav_sample)
                _, folder_id = os.path.split(_dir)
                
                mic_id = 'GPAS'
                folder_id = folder_id.replace('_MEMS', '_GPAS')
                file_id = file_id.replace('_MEMS', '_GPAS_ch*')
                
                wav_sample = os.path.join(args.data_folder, mic_id, folder_id, file_id)
                
                file_ids = sorted(glob(wav_sample))
                
                data_list = []
                
                #file_ids[0] -> ch1
                #file_ids[1] -> ch2
                #file_ids[2] -> ch3
                #file_ids[3] -> ch4
                #file_ids[4] -> ch5
                #file_ids[5] -> ch6
                #file_ids[6] -> ch7
                
                if self.ch == 1:
                    data_list.append(file_ids[0])
                elif self.ch == 2:
                    data_list.append(file_ids[0]) # 1
                    data_list.append(file_ids[3]) # 
                elif self.ch == 3:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                elif self.ch == 4:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                elif self.ch == 5:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                    data_list.append(file_ids[5])
                elif self.ch == 6:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                    data_list.append(file_ids[5])
                    data_list.append(file_ids[6])
                elif self.ch == 7:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[1])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                    data_list.append(file_ids[5])
                    data_list.append(file_ids[6])
                
                processed_data = get_individual_samples_torchaudio_multi_channel(args, data_list, self.sample_rate, label)
                
            else:
                wav_sample = os.path.join(args.data_folder, data)
                processed_data = get_individual_samples_torchaudio(args, wav_sample, self.sample_rate, label)
            data_set = (processed_data[0], processed_data[1])
            self.audio_data.append(data_set)
            self.class_nums[processed_data[1]] += 1
        
        print(len(self.audio_data))
        
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('total number of audio data: {}'.format(len(self.audio_data)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
                
        if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
            from transformers import AutoFeatureExtractor
            self.speech_extractor = AutoFeatureExtractor.from_pretrained(args.model)
        
        self.audio_images = []
        
        for index in range(len(self.audio_data)): #for the training set, 4142
            audio, label = self.audio_data[index][0], self.audio_data[index][1] # wav, label
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
                    image = audio
                    if args.framework == 'transformers':
                        inputs = self.speech_extractor(audio, sampling_rate=self.sample_rate)
                        image = torch.from_numpy(inputs['input_values'][0])
                else:
                    image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                audio_image.append(image)
            self.audio_images.append((audio_image, label))
            

    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        audio_image = audio_images[0]
        return audio_image, label

    def __len__(self):
        return len(self.audio_images)
        
        
class EmoCREMADataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        
        print('train_flag', train_flag)
        
        if train_flag:
            annotation_file = os.path.join(args.data_folder, args.train_annotation_file)
        else:
            annotation_file = os.path.join(args.data_folder, args.test_annotation_file)
        
        
        self.train_flag = train_flag
        self.args = args
        self.ch = self.args.channel

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels

        """ get dataset information """        
        df = pd.read_csv(annotation_file)        
        files = df['file_path'].values.tolist()
        labels = df['label'].values.tolist()
        
        self.filenames = []
        self.audio_data = []  # each sample will be a tuple with (audio_data, label)

        if print_flag:
            print('*' * 20)  
            print("Extracting individual emotional data samples..")
        
        self.class_nums = np.zeros(args.n_cls)
        
        '''
        ch1: ch1
        ch2: ch1, ch4
        ch3: ch1, ch4, ch5
        ch4: ch1, ch3, ch4, ch5
        ch5: ch1, ch3, ch4, ch5, ch6
        ch6: ch1, ch3, ch4, ch5, ch6, ch7        
        '''
        
        if self.args.test_origin:
            for i, (data, label) in enumerate(zip(files, labels)):
                if self.train_flag:
                    if self.ch > 0:
                        file_id = os.path.basename(data)
                        wav_samples = os.path.join(args.data_folder, 'crema_gpas', file_id.replace('.wav', '_ch*'))
                        
                        file_ids = sorted(glob(wav_samples))
                        
                        data_list = []
                        
                        
                        #file_ids[0] -> ch1
                        #file_ids[1] -> ch2
                        #file_ids[2] -> ch3
                        #file_ids[3] -> ch4
                        #file_ids[4] -> ch5
                        #file_ids[5] -> ch6
                        #file_ids[6] -> ch7
                        
                        if self.ch == 1:
                            data_list.append(file_ids[0])
                        elif self.ch == 2:
                            data_list.append(file_ids[0]) # 1
                            data_list.append(file_ids[3]) # 
                        elif self.ch == 3:
                            data_list.append(file_ids[0])
                            data_list.append(file_ids[3])
                            data_list.append(file_ids[4])
                        elif self.ch == 4:
                            data_list.append(file_ids[0])
                            data_list.append(file_ids[2])
                            data_list.append(file_ids[3])
                            data_list.append(file_ids[4])
                        elif self.ch == 5:
                            data_list.append(file_ids[0])
                            data_list.append(file_ids[2])
                            data_list.append(file_ids[3])
                            data_list.append(file_ids[4])
                            data_list.append(file_ids[5])
                        elif self.ch == 6:
                            data_list.append(file_ids[0])
                            data_list.append(file_ids[2])
                            data_list.append(file_ids[3])
                            data_list.append(file_ids[4])
                            data_list.append(file_ids[5])
                            data_list.append(file_ids[6])
                        elif self.ch == 7:
                            data_list.append(file_ids[0])
                            data_list.append(file_ids[1])
                            data_list.append(file_ids[2])
                            data_list.append(file_ids[3])
                            data_list.append(file_ids[4])
                            data_list.append(file_ids[5])
                            data_list.append(file_ids[6])
                        processed_data = get_individual_samples_torchaudio_multi_channel(args, data_list, self.sample_rate, label)
                    else:
                        wav_sample = os.path.join(args.data_folder, data)
                        processed_data = get_individual_samples_torchaudio(args, wav_sample, self.sample_rate, label)
                else:                    
                    wav_sample = os.path.join(args.data_folder, data)
                    processed_data = get_individual_samples_torchaudio(args, wav_sample, self.sample_rate, label)
                    
                    if self.args.mic == 'mems':
                        file_id = os.path.basename(data)
                        wav_sample = os.path.join(args.data_folder, 'crema_mems', file_id)
                    elif self.args.mic == 'gpas':
                        file_id = os.path.basename(data)
                        wav_sample = os.path.join(args.data_folder, 'crema_gpas', file_id)
                    else:
                        wav_sample = os.path.join(args.data_folder, data)
                
            
        else:
        
            for i, (data, label) in enumerate(zip(files, labels)):
                if self.ch > 0:
                    file_id = os.path.basename(data)
                    wav_samples = os.path.join(args.data_folder, 'crema_gpas', file_id.replace('.wav', '_ch*'))
                    
                    file_ids = sorted(glob(wav_samples))
                    
                    data_list = []
                    
                    #file_ids[0] -> ch1
                    #file_ids[1] -> ch2
                    #file_ids[2] -> ch3
                    #file_ids[3] -> ch4
                    #file_ids[4] -> ch5
                    #file_ids[5] -> ch6
                    #file_ids[6] -> ch7
                    
                    if self.ch == 1:
                        data_list.append(file_ids[0])
                    elif self.ch == 2:
                        data_list.append(file_ids[0]) # 1
                        data_list.append(file_ids[3]) # 
                    elif self.ch == 3:
                        data_list.append(file_ids[0])
                        data_list.append(file_ids[3])
                        data_list.append(file_ids[4])
                    elif self.ch == 4:
                        data_list.append(file_ids[0])
                        data_list.append(file_ids[2])
                        data_list.append(file_ids[3])
                        data_list.append(file_ids[4])
                    elif self.ch == 5:
                        data_list.append(file_ids[0])
                        data_list.append(file_ids[2])
                        data_list.append(file_ids[3])
                        data_list.append(file_ids[4])
                        data_list.append(file_ids[5])
                    elif self.ch == 6:
                        data_list.append(file_ids[0])
                        data_list.append(file_ids[2])
                        data_list.append(file_ids[3])
                        data_list.append(file_ids[4])
                        data_list.append(file_ids[5])
                        data_list.append(file_ids[6])
                    elif self.ch == 7:
                        data_list.append(file_ids[0])
                        data_list.append(file_ids[1])
                        data_list.append(file_ids[2])
                        data_list.append(file_ids[3])
                        data_list.append(file_ids[4])
                        data_list.append(file_ids[5])
                        data_list.append(file_ids[6])
                    
                    processed_data = get_individual_samples_torchaudio_multi_channel(args, data_list, self.sample_rate, label)
                    
                else:
                    wav_sample = os.path.join(args.data_folder, data)
                    processed_data = get_individual_samples_torchaudio(args, wav_sample, self.sample_rate, label)
        
        
        data_set = (processed_data[0], processed_data[1])
        self.audio_data.append(data_set)
        self.class_nums[processed_data[1]] += 1
        
        
        
        for i, (data, label) in enumerate(zip(files, labels)):
            if self.ch > 0:
                file_id = os.path.basename(data)
                wav_samples = os.path.join(args.data_folder, 'crema_gpas', file_id.replace('.wav', '_ch*'))
                
                file_ids = sorted(glob(wav_samples))
                
                data_list = []
                
                #file_ids[0] -> ch1
                #file_ids[1] -> ch2
                #file_ids[2] -> ch3
                #file_ids[3] -> ch4
                #file_ids[4] -> ch5
                #file_ids[5] -> ch6
                #file_ids[6] -> ch7
                
                if self.ch == 1:
                    data_list.append(file_ids[0])
                elif self.ch == 2:
                    data_list.append(file_ids[0]) # 1
                    data_list.append(file_ids[3]) # 
                elif self.ch == 3:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                elif self.ch == 4:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                elif self.ch == 5:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                    data_list.append(file_ids[5])
                elif self.ch == 6:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                    data_list.append(file_ids[5])
                    data_list.append(file_ids[6])
                elif self.ch == 7:
                    data_list.append(file_ids[0])
                    data_list.append(file_ids[1])
                    data_list.append(file_ids[2])
                    data_list.append(file_ids[3])
                    data_list.append(file_ids[4])
                    data_list.append(file_ids[5])
                    data_list.append(file_ids[6])
                
                processed_data = get_individual_samples_torchaudio_multi_channel(args, data_list, self.sample_rate, label)
                
            else:
                if self.args.mic == 'mems':
                    file_id = os.path.basename(data)
                    wav_sample = os.path.join(args.data_folder, 'crema_mems', file_id)
                elif self.args.mic == 'gpas':
                    file_id = os.path.basename(data)
                    wav_sample = os.path.join(args.data_folder, 'crema_gpas', file_id)
                else:
                    wav_sample = os.path.join(args.data_folder, data)
                
                processed_data = get_individual_samples_torchaudio(args, wav_sample, self.sample_rate, label)
            data_set = (processed_data[0], processed_data[1])
            self.audio_data.append(data_set)
            self.class_nums[processed_data[1]] += 1
        
        print(len(self.audio_data))
        
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('total number of audio data: {}'.format(len(self.audio_data)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
                
        if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
            from transformers import AutoFeatureExtractor
            self.speech_extractor = AutoFeatureExtractor.from_pretrained(args.model)
        
        self.audio_images = []
        
        for index in range(len(self.audio_data)): #for the training set, 4142
            audio, label = self.audio_data[index][0], self.audio_data[index][1] # wav, label
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if args.model in ['facebook/hubert-base-ls960', 'facebook/wav2vec2-base', 'microsoft/wavlm-base-plus']:
                    image = audio
                    if args.framework == 'transformers':
                        inputs = self.speech_extractor(audio, sampling_rate=self.sample_rate)
                        image = torch.from_numpy(inputs['input_values'][0])
                else:
                    image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                audio_image.append(image)
            self.audio_images.append((audio_image, label))
            

    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        audio_image = audio_images[0]
        return audio_image, label

    def __len__(self):
        return len(self.audio_images)
        