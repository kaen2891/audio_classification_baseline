import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, HubertModel

class PretrainedSpeechModels(nn.Module): # using pretrained speech models from huggingface
    def __init__(self, pretrained_model, pretrained_name, final_feat_dim):
        super().__init__()

        self.pretrained_name = pretrained_name
        self.feature_extractors = pretrained_model.from_pretrained(self.pretrained_name)
        self.final_feat_dim = final_feat_dim
        
    def forward(self, x, y=None, y2=None, da_index=None, patch_mix=False, time_domain=False, args=None, alpha=None, training=False):
        print('input x', x.size())
        x = self.feature_extractors(x)
        x = x["last_hidden_state"].mean(dim=1)
        return x  
