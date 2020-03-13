import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class NASEncoderLayer(nn.Module):
    def __init__(self, args, layer, hidden_size, dropout):
        super().__init__()
        self.args = args
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class NASDecoderLayer(nn.Module):
    def __init__(self, args, layer, hidden_size, dropout):
        super().__init__()
        self.args = args
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_DECODER[layer](hidden_size, dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

    def clear_buffer(self, *args):
        return self.op.clear_buffer(*args)

    def set_buffer(self, *args):
        return self.op.set_buffer(*args)


class NASEncoder(nn.Module):
    def __init__(self, args, arch, embed_tokens):
        super().__init__()
        self.args = args
        self.arch = arch
        self.num_layers = args.enc_layers
        self.hidden_size = args.hidden_size
        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        embed_dim = embed_tokens.embedding_dim
        self.dropout = args.dropout
        self.embed_scale = math.sqrt(embed_dim)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx, 
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            NASEncoderLayer(args, self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(embed_dim)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        positions = self.embed_positions(src_tokens)
        x = embed + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens):
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        #if not encoder_padding_mask.any():
        #    encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        x = self.layer_norm(x)
        return {
            'encoder_out': x, # T x B x C
            'encoder_padding_mask': encoder_padding_mask, # B x T
            'encoder_embedding': encoder_embedding, # B x T x C
        }


class NASDecoder(nn.Module):
    def __init__(self, args, arch, padding_idx=0):
        super().__init__()
        self.args = args
        self.arch = arch
        self.num_layers = args.dec_layers
        self.hidden_size = args.hidden_size
        self.prenet_hidden_size = args.prenet_hidden_size
        self.padding_idx = padding_idx
        self.dropout = args.dropout
        self.in_dim = args.audio_num_mel_bins
        self.out_dim = args.audio_num_mel_bins + 1
        self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size, self.padding_idx,
            init_size=self.max_target_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            NASDecoderLayer(args, self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.hidden_size)
        self.project_out_dim = Linear(self.hidden_size, self.out_dim, bias=False)
        self.prenet_fc1 = Linear(self.in_dim, self.prenet_hidden_size)
        self.prenet_fc2 = Linear(self.prenet_hidden_size, self.prenet_hidden_size)
        self.prenet_fc3 = Linear(self.prenet_hidden_size, self.hidden_size, bias=False)

    def forward_prenet(self, x):
        mask = x.abs().sum(-1, keepdim=True).ne(0).float()
        x = self.prenet_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=True)
        x = self.prenet_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=True)
        x = self.prenet_fc3(x)
        x = F.relu(x)
        x = x * mask
        return x

    def forward(
        self,
        prev_output_mels, # B x T x 80
        encoder_out=None, # T x B x C
        encoder_padding_mask=None, # B x T x C
        target_mels=None,
        incremental_state=None,
    ):
        # embed positions
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_output_mels.abs().sum(-1),
                incremental_state=incremental_state
            )
            prev_output_mels = prev_output_mels[:, -1:, :]
            positions = positions[:, -1:, :]
            # compute padding mask
            #self_attn_padding_mask = prev_output_mels.abs().sum(-1).eq(self.padding_idx)
            #self_attn_padding_mask[:, 0] = False
            #if not self_attn_padding_mask.any():
            #    self_attn_padding_mask = None
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                target_mels.abs().sum(-1),
                incremental_state=incremental_state
            )
            # compute padding mask
            #self_attn_padding_mask = target_mels.abs().sum(-1).eq(self.padding_idx)
            #if not self_attn_padding_mask.any():
            #    self_attn_padding_mask = None
            self_attn_padding_mask = None

        # convert mels through prenet
        x = self.forward_prenet(prev_output_mels)
        # embed positions
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        all_attn_logits = []

        # decoder layers
        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask
            ) 
            all_attn_logits.append(attn_logits)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        
        # B x T x C -> B x T x 81
        x = self.project_out_dim(x)

        #attn_logits = torch.stack(all_attn_logits, dim=1) # B x n_layers x head x target_len x src_len
        return x, all_attn_logits

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class NASNetwork(nn.Module):
    def __init__(self, args, arch, dictionary):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch =arch
        self.enc_layers = args.enc_layers
        self.dec_layers = args.dec_layers
        self.enc_arch = self.arch[:self.enc_layers]
        self.dec_arch = self.arch[self.enc_layers:self.enc_layers+self.dec_layers]
        self.hidden_size = args.hidden_size
        self.mel = args.audio_num_mel_bins
        self.build_model()

    def build_model(self):
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            self.padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
            return emb

        self.encoder_embed_tokens = build_embedding(self.dictionary, self.hidden_size)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        return NASEncoder(self.args, self.enc_arch,  self.encoder_embed_tokens)

    def build_decoder(self):
        return NASDecoder(self.args, self.dec_arch, padding_idx=self.padding_idx)

    def forward_encoder(self, src_tokens, *args, **kwargs):
        return self.encoder(src_tokens)

    def forward_decoder(self, prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=None, *args, **kwargs):
        decoder_output, attn_logits = self.decoder(
            prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=incremental_state)
        return decoder_output, attn_logits

    def forward(self, src_tokens, prev_output_mels, target_mels, *args, **kwargs):
        encoder_outputs = self.encoder(src_tokens)
        encoder_out = encoder_outputs['encoder_out']
        encoder_padding_mask = encoder_outputs['encoder_padding_mask']
        decoder_output, attn_logits = self.decoder(prev_output_mels, encoder_out, encoder_padding_mask, target_mels=target_mels)
        return decoder_output, attn_logits


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

