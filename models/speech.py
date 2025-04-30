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
        #print('input x', x.size())

        T = int(args.T * args.sample_rate)
        total_frames = x.size(1)
        
        # overlap ratio (e.g., 0.5 = 50% overlap)
        overlap = args.overlap
        stride = int(T * overlap)
        
        # calculate num_chunks based on stride
        num_chunks = (total_frames - T) // stride + 1
        
        #print('input x', x.size())
        #print('total_frames {} T {} stride {}'.format(total_frames, T, stride))
        #print('num_chunks', num_chunks)
        
        outputs = []
        
        for i in range(num_chunks):
            start = int(i * stride)
            end = int(start + T)
            #print('start {} end {}'.format(start, end))
            
            if start >= total_frames:
                break
        
            if end > total_frames:
                x_chunk = x[:, start:total_frames]  # (8, remained_length)
                pad_length = end - total_frames
                #print(f"Padding {pad_length} frames")
        
                # Pad: (batch, time)
                x_chunk = F.pad(x_chunk, (0, 0, 0, pad_length))  # (left, right, top, bottom)
                # shape : (8, T)
            else:
                x_chunk = x[:, start:end]  # (8, T)
            
        
            if end > total_frames:
                
                break
        
            x_chunk = x[:, start:end]  # (8, T)
            #print('{}th start {} end {} x_chunk {}'.format(i, start, end, x_chunk.size()))
            
            out_chunk = self.feature_extractors(x_chunk)  # (8, 100, 768)
            
            pooled_chunk = out_chunk["last_hidden_state"].mean(dim=1)  # (8, 768)
            #print('{}th pooled_chunk {}'.format(i, pooled_chunk.size()))
            
            outputs.append(pooled_chunk)
        final_output = torch.stack(outputs, dim=1) # Batch, num_chunks, 768
        #print('final_output', final_output.size())
        
        output, (h_n, c_n) = self.LSTM(final_output)
        output = output[:, -1, :]
        #print('output', output.size()) # Batch, 768
        return output
        
