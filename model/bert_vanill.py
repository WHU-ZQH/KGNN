import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from torch.nn.utils import weight_norm
import torch
from torch import nn

class BERT_vanilla(nn.Module):
    def __init__(self, bert, args):
        super(BERT_vanilla, self).__init__()
        self.args = args
        self.bert = bert
        self.hid_num = 768

        self.fc_out = nn.Linear(self.hid_num, 3)
        self.text_embed_dropout = nn.Dropout(args.dropout_rate)
        self.squeezeEmbedding = SqueezeEmbedding()

    def forward(self, feature):
        feature = feature.long()
        text_len = torch.sum(feature != 0, dim=-1).cpu()
                                                                                                
        if self.args.is_bert ==1 :                                                                                                                                      
            text, text_pool = self.bert(feature, output_all_encoded_layers=False)                                                                                               
            text = self.text_embed_dropout(text_pool)                                                                                                                        
        elif self.args.is_bert ==2:                                                                                                                                     
            text=self.bert(feature).last_hidden_state                                                                                                                   
                                                                                                                                 
        output=self.fc_out(text)
        return output