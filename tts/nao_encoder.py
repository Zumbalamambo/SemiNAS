import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 length,
                 source_length,
                 emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 dropout,
                 ):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.dropout = dropout
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.mlp = nn.ModuleList([])
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.append(nn.Linear(self.hidden_size, self.mlp_hidden_size))
            elif i == self.mlp_layers - 1:
                self.mlp.append(nn.Linear(self.mlp_hidden_size, self.hidden_size))
            else:
                self.mlp.append(nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size))
        self.regressor = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        out, hidden = self.rnn(x)
        out = self.out_proj(out)
        out += x
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        encoder_hidden = hidden
        
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        arch_emb = out
        
        residual = out
        for i, mlp_layer in enumerate(self.mlp):
            out = mlp_layer(out)
            out = F.relu(out)
            if i != self.mlp_layers:
                out = F.dropout(out, self.dropout, training=self.training)
        out = (residual + out) * math.sqrt(0.5)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value
    
    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb