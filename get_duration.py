# -*- coding: cp949 -*-
from glob import glob
import torchaudio
import os
import numpy as np
#data = '/data2/wkuData/가을문단all/'

all_wav = sorted(glob('/data2/wkuData/가을문단all/*.wav'))



def analyze_numbers(numbers, n):
    if not numbers:
        return None, []
    
    # 평균 계산
    mean_value = sum(numbers) / len(numbers)
    
    # 숫자 크기 기준으로 정렬 (내림차순)
    sorted_by_value = sorted(numbers, reverse=True)
    
    # 상위 n개 뽑기
    top_n_largest = sorted_by_value[:n]
    
    return mean_value, top_n_largest

def check_len(data_list, length):
    output_list = []
    for data in data_list:
        if data > length:
            output_list.append(data)
    return output_list

data_all = []
for i, data in enumerate(all_wav):
    y, sr = torchaudio.load(data)
    
    #print('{}th y {} sr {} len {}'.format(i, y.size(), sr, y.size(-1)/sr))
    data_all.append(y.size(-1)/sr)

n = 100
mean, longest_numbers = analyze_numbers(data_all, n)




print("평균:", mean)
print(f"자릿수가 가장 긴 수 {n}개:", longest_numbers)

len_check = check_len(data_all, 180)
print('len_check', len_check, len(len_check))
