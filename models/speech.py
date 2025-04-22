import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, HubertModel

class PretrainedSpeechModels(nn.Module): # using pretrained speech models from huggingface
    def __init__(self, pretrained_model, pretrained_name, final_feat_dim, num_lstm, bidirectional=False):
    #def __init__(self, pretrained_model, pretrained_name, final_feat_dim, num_lstm):
        super().__init__()

        self.pretrained_name = pretrained_name
        self.feature_extractors = pretrained_model.from_pretrained(self.pretrained_name)
        self.final_feat_dim = final_feat_dim
        self.num_lstm = num_lstm
        self.bidirctional = bidirectional
        self.avgpool = nn.AvgPool2d(kernel_size=3, padding=1, stride=(1,2))
        '''
        if self.bidirectional:   
            self.LSTM = nn.LSTM(self.final_feat_dim, self.final_feat_dim//2, self.num_lstm, batch_first=True, bidirectional=True) # num_layer=3
        else:
        '''
        self.LSTM = nn.LSTM(self.final_feat_dim, self.final_feat_dim, self.num_lstm, batch_first=True, bidirectional=False) # num_layer=3
        
    def forward(self, x, y=None, y2=None, da_index=None, patch_mix=False, time_domain=False, args=None, alpha=None, training=False):
        print('input x', x.size())
        
        args.T = 500
        total_frames = x.size(1)
        num_chunks = total_frames // T
        
        print('total_frames {} num_chunks {}'.format(total_frames, num_chunks))
                
        outputs = []
        
        for i in range(num_chunks):
            start = i * T
            end = start + T
            x_chunk = x[:, start:end, :] # 8, T, 768
            
            out_chunk = self.feature_extractors(x_chunk) # 8, 100, 768
            
            pooled_chunk = out_chunk["last_hidden_state"].mean(dim=1)
            
            outputs.append(pooled_chunk)
        
        final_output = torch.stack(outputs, dim=1)
        
        output, (h_n, c_n) = self.LSTM(final_output)
        output = output[:, -1, :]
        print('output', output.size())
        exit()
        return output
        
        iter_num = spec_length // hop_num
        
        
        x = self.feature_extractors(x)
        x = x["last_hidden_state"].mean(dim=1)
        return x  
