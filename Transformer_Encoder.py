import torch
import torch.nn as nn
import numpy as np
import math

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
# input_window = 100 # number of input steps
# output_window = 1 # number of prediction steps, in this model its fixed to one
# batch_size = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   # torch.Size([max_len, 1, d_model])
        #pe.requires_grad = False
        self.register_buffer('pe', pe) ## 매개변수로 간주하지 않기 위한 것

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class Transformer(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1,batch_size=128,x_frames = 20):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()
        self.batch_size = batch_size
        self.x_frames = x_frames
        self.output_linear = nn.Linear(self.x_frames, 1)

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        output = self.output_linear(output.view(self.batch_size, -1))

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

#
#
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)   # torch.Size([max_len, 1, d_model])
#         #pe.requires_grad = False
#         self.register_buffer('pe', pe) ## 매개변수로 간주하지 않기 위한 것
#
#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# class TransAm(nn.Module):
#     def __init__(self, feature_size, dropout):
#         super(TransAm, self).__init__()
#         self.model_type = 'Transformer'
#
#         self.src_mask = None
#         self.tgt_mask = None
#         self.pos_encoder = PositionalEncoding(feature_size)
#         # self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
#         # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#
#         # d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
#         # num_decoder_layers: int = 6,
#         # d_model > 임베딩 차원
#         ## dim_feedforward: int = 2048 >> feed forward linear layer 의 차원
#         self.transformer = nn.Transformer(d_model=feature_size, nhead=4, dropout=dropout,
#                                           num_encoder_layers=1,num_decoder_layers=1)
#         self.decoder = nn.Linear(feature_size, 1)
#
#         self.linear = nn.Linear(feature_size, 1)
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, src, tgt):
#
#         tgt = tgt.unsqueeze(0).to(torch.float32)
#
#         # torch.Size([20, 128, 1])
#         # torch.Size([128, 1])
#         if self.src_mask is None or self.src_mask.size(0) != len(src):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(src)).to(device)
#             self.src_mask = mask
#
#         if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
#             self.tgt_mask = mask
#
#
#         src = self.pos_encoder(src)
#         tgt = self.pos_encoder(tgt)
#
#
#         # torch.Size([20, 128, 16])
#         # torch.Size([1, 128, 16])
#         # output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
#         # output = self.decoder(output)
#         output = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask = self.tgt_mask)
#
#         return self.linear(output)
#
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)                ## 하삼각행렬
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
