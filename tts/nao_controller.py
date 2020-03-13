import logging
import copy
import numpy as np
import utils
import torch
from torch import nn
import torch.nn.functional as F
from nao_encoder import Encoder
from nao_decoder import Decoder


class NAODataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
        super(NAODataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = copy.deepcopy(inputs)
        self.targets = copy.deepcopy(targets)
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = [self.targets[index]]

        if self.train:
            decoder_input = [self.sos_id] + encoder_input[:-1]
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'encoder_target': torch.FloatTensor(encoder_target),
                'decoder_input': torch.LongTensor(decoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
        else:
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
            if encoder_target is not None:
                sample['encoder_target'] = torch.FloatTensor(encoder_target)
        return sample
    
    def __len__(self):
        return len(self.inputs)


class NAO(nn.Module):
    def __init__(self, args):
        super(NAO, self).__init__()
        self.vocab_size = args.controller_vocab_size
        self.hidden_size = args.controller_hidden_size
        self.encoder_layers = args.controller_encoder_layers
        self.encoder_length = args.controller_encoder_length
        self.source_length = args.controller_source_length
        self.mlp_layers = args.controller_mlp_layers
        self.mlp_hidden_size = args.controller_mlp_hidden_size
        self.decoder_layers = args.controller_decoder_layers
        self.dropout = args.controller_dropout
        self.decoder_length = args.controller_decoder_length
        self.lr = args.controller_lr
        self.weight_decay = args.controller_weight_decay
        self.clip_grad_norm = args.controller_clip_grad_norm
        self.batch_size = args.controller_batch_size
        self.trade_off = args.controller_trade_off

        self.build_model()
        self.optimizer = self.build_optimizer()
        self.datasets = {}
        self.queues = {}
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def build_model(self):
        self.encoder = Encoder(
            self.encoder_layers,
            self.vocab_size,
            self.hidden_size,
            self.encoder_length,
            self.source_length,
            self.hidden_size,
            self.mlp_layers,
            self.mlp_hidden_size,
            self.dropout,
        )
        self.decoder = Decoder(
            self.decoder_layers,
            self.vocab_size,
            self.hidden_size,
            self.decoder_length,
            self.encoder_length,
            self.dropout,
        )

        self.flatten_parameters()
        if torch.cuda.is_available():
            self.cuda()

    def build_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def build_dataset(self, split, inputs, targets, train):
        self.datasets[split] = NAODataset(inputs, targets, train)

    def build_queue(self, split, batch_size=None, shuffle=True, pin_memory=False):
        if batch_size is None:
            batch_size = self.batch_size
        self.queues[split] = torch.utils.data.DataLoader(self.datasets[split], batch_size, shuffle=shuffle, pin_memory=pin_memory)

    def generate_synthetic_data(self, exclude=[], maxn=1000):
        synthetic_encoder_input = []
        synthetic_encoder_target = []
        while len(synthetic_encoder_input) < maxn:
            synthetic_arch = utils.generate_arch(1, self.source_length, self.vocab_size-1)[0]
            synthetic_arch = utils.parse_arch_to_seq(synthetic_arch)
            if synthetic_arch not in exclude and synthetic_arch not in synthetic_encoder_input:
                synthetic_encoder_input.append(synthetic_arch)
    
        nao_synthetic_dataset = NAODataset(synthetic_encoder_input, None, False)      
        nao_synthetic_queue = torch.utils.data.DataLoader(nao_synthetic_dataset, batch_size=len(nao_synthetic_dataset), shuffle=False, pin_memory=True)

        self.eval()
        with torch.no_grad():
            for sample in nao_synthetic_queue:
                encoder_input = sample['encoder_input'].cuda()
                _, _, _, predict_value = self.encoder(encoder_input)
                synthetic_encoder_target += predict_value.data.squeeze().tolist()
        assert len(synthetic_encoder_input) == len(synthetic_encoder_target)
        return synthetic_encoder_input, synthetic_encoder_target

    def train_epoch(self, split):
        objs = utils.AvgrageMeter()
        mse = utils.AvgrageMeter()
        nll = utils.AvgrageMeter()
        queue = self.queues[split]
        self.train()
        for step, sample in enumerate(queue):
            encoder_input = utils.move_to_cuda(sample['encoder_input'])
            encoder_target = utils.move_to_cuda(sample['encoder_target'])
            decoder_input = utils.move_to_cuda(sample['decoder_input'])
            decoder_target = utils.move_to_cuda(sample['decoder_target'])
            
            self.optimizer.zero_grad()
            predict_value, log_prob, arch = self(encoder_input, decoder_input)
            loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
            loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
            loss = self.trade_off * loss_1 + (1 - self.trade_off) * loss_2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            
            n = encoder_input.size(0)
            objs.update(loss.data, n)
            mse.update(loss_1.data, n)
            nll.update(loss_2.data, n)
        return objs.avg, mse.avg, nll.avg

    def infer(self, split, step, direction='+'):
        queue = self.queues[split]
        new_arch_list = []
        self.eval()
        for i, sample in enumerate(queue):
            encoder_input = utils.move_to_cuda(sample['encoder_input'])
            self.zero_grad()
            new_arch = self.generate_new_arch(encoder_input, step, direction=direction)
            new_arch_list.extend(new_arch.data.squeeze().tolist())
        return new_arch_list
    
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return predict_value, decoder_outputs, archs
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='+'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, new_archs = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        return new_archs
