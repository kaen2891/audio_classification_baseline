from glob import glob
import torchaudio


all_data = sorted(glob('./data/TESS_TDMS_to_CSV_output_MEMS_FINAL_WAV/*/*.wav'))


all_len = []
for i, data in enumerate(all_data):
    y, sr = torchaudio.load(data)
    print(sr)
    
    wav_len = y.size(-1)/sr
    print(wav_len)
    
    all_len.append(wav_len)
    
import numpy as np
print(np.mean(all_len))
print(np.max(all_len))
print(np.min(all_len))
    