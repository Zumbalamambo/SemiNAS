import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import utils
import logging
from text_encoder import TokenTextEncoder
from preprocessor import _process_utterance
import audio
import model, model_ws


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    id = torch.LongTensor([s['id'] for s in samples])
    utt_id = torch.LongTensor([s['utt_id'] for s in samples]) if samples[0]['utt_id'] is not None else None
    text = [s['text'] for s in samples] if samples[0]['text'] is not None else None
    src_tokens = utils.collate_tokens([s['source'] for s in samples], pad_idx)
    target = utils.collate_mels([s['target'] for s in samples], pad_idx) if samples[0]['target'] is not None else None
    prev_output_mels = utils.collate_mels([s['target'] for s in samples], pad_idx, shift_right=True) if target is not None else None
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    target_lengths = torch.LongTensor([s['target'].shape[0] for s in samples]) if target is not None else None
    if target is not None and target_lengths is not None:
        target_lengths, sort_order = target_lengths.sort(descending=True)
        target = target.index_select(0, sort_order)
        prev_output_mels = prev_output_mels.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        src_lengths = src_lengths.index_select(0, sort_order)
    else:
        src_lengths, sort_order = src_lengths.sort(descending=True)    
        src_tokens = src_tokens.index_select(0, sort_order)
    id = id.index_select(0, sort_order)
    utt_id = utt_id.index_select(0, sort_order) if utt_id is not None else None
    text = [text[i] for i in sort_order] if text is not None else None
    ntokens = sum(len(s['source']) for s in samples)
    nmels = sum(len(s['target']) for s in samples) if target is not None else None
    
    batch = {
        'id': id,
        'utt_id': utt_id,
        'nsamples': len(samples),
        'ntokens': ntokens,
        'nmels': nmels,
        'text': text,
        'src_tokens': src_tokens,
        'src_lengths': src_lengths,
        'targets': target,
        'target_lengths': target_lengths,
        'prev_output_mels': prev_output_mels
    }
    return batch


class LJSpeechRawDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, model_hparams, phone_encoder, utt_ids=None, shuffle=False):
        super().__init__()
        self.model_hparams = model_hparams
        self.phone_encoder = phone_encoder
        self.shuffle = shuffle

        self.utt_ids = None
        self.texts = None
        self.phones = None
        self.mels = None
        self.sizes = None       
        self.read_data(data_dir, utt_ids)

    def produce_result(self, utt_id, text, phone, mel):
        phones = phone.split(" ")
        phones += ['|']
        phones = ' '.join(phones)
        try:
            utt_id = int(utt_id)
            phone_encoded = torch.LongTensor(self.phone_encoder.encode(phones) + [self.phone_encoder.eos()])
            mel = torch.Tensor(mel.T) #(T*80)
        except Exception as e:
            logging.info('{} {}'.format(e, text))
            return None

        return utt_id, text, phone_encoded, mel

    def read_data(self, data_dir, utt_ids=None):
        data_df = pd.read_csv(os.path.join(data_dir, 'metadata_phone.csv'))
        self.utt_ids = []
        self.texts = []
        self.phones = []
        self.mels = []
        self.sizes = []     
        if utt_ids is None:
            for idx, r in data_df.iterrows():
                utt_id, text, phone, mel = self.produce_result(idx, r['txt2'], r['phone2'], self.process_wav(data_dir, r['wav'])[1])
                self.utt_ids.append(utt_id)
                self.texts.append(text)
                self.phones.append(phone)
                self.mels.append(mel)
                self.sizes.append(len(mel))
        else:
            for utt_id in utt_ids:
                r = data_df.iloc[utt_id]
                utt_id, text, phone, mel = self.produce_result(utt_id, r['txt2'], r['phone2'], self.process_wav(data_dir, r['wav'])[1])
                self.utt_ids.append(utt_id)
                self.texts.append(text)
                self.phones.append(phone)
                self.mels.append(mel)
                self.sizes.append(len(mel))
        
    def __getitem__(self, index):
        sample = {"id": index, 
                "utt_id": self.utt_ids[index] if self.utt_ids is not None else None,
                "text":   self.texts[index] if self.texts is not None else None,
                "source": self.phones[index] if self.phones is not None else None,
                "target": self.mels[index] if self.mels is not None else None}
        return sample

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.phone_encoder.pad(), eos_idx=self.phone_encoder.eos()
        )

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self.sizes[index], self.model_hparams.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(np.array(self.sizes)[indices], kind='mergesort')]


class LJSpeechDataset(LJSpeechRawDataset):
    def __init__(self, data_dir, prefix, model_hparams, phone_encoder, shuffle=False):
        super().__init__(data_dir, model_hparams, phone_encoder, prefix, shuffle)

    def read_data(self, data_dir, prefix):
        if os.path.exists(os.path.join(data_dir, '{}.idx'.format(prefix))):
            with open(os.path.join(data_dir, '{}.idx'.format(prefix)), 'r') as f:
                self.utt_ids = [int(line) for line in f.readlines()]
        if os.path.exists(os.path.join(data_dir, '{}.text'.format(prefix))):
            with open(os.path.join(data_dir, '{}.text'.format(prefix)), 'r') as f:
                self.texts = [line.strip() for line in f.readlines()]
        if os.path.exists(os.path.join(data_dir, '{}.phone'.format(prefix))):
            with open(os.path.join(data_dir, '{}.phone'.format(prefix)), 'r') as f:
                self.phones = [torch.LongTensor(list(map(int, line.strip().split()))) for line in f.readlines()]
        if os.path.exists(os.path.join(data_dir, '{}.mel'.format(prefix))):
            with open(os.path.join(data_dir, '{}.mel'.format(prefix)), 'rb') as f:
                mels = np.load(f, allow_pickle=True)
            self.mels = [torch.Tensor(mel) for mel in mels]

        if self.mels:
            self.sizes = [len(mel) for mel in self.mels]
        else:
            self.sizes =[len(text) for text in self.texts]
        

def set_ljspeech_hparams(model_hparams):
    model_hparams.max_sample_size = 1500
    model_hparams.symbol_size = None
    model_hparams.save_npz = False
    model_hparams.audio_num_mel_bins = 80
    model_hparams.audio_sample_rate = 22050
    model_hparams.num_freq = 513  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    model_hparams.hop_size = 256  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    model_hparams.win_size = 1024  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    model_hparams.fmin = 0  # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    model_hparams.fmax = 8000  # To be increased/reduced depending on data.
    model_hparams.n_fft = 1024  # Extra window size is filled with 0 paddings to match this parameter
    model_hparams.min_level_db = -100
    model_hparams.ref_level_db = 20
    # Griffin Lim
    model_hparams.power = 1.5  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    model_hparams.magnitude_power = 1  # The power of the spectrogram magnitude (1. for energy, 2. for power)
    model_hparams.griffin_lim_iters = 60  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.

    # #M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    model_hparams.trim_fft_size = 512
    model_hparams.trim_hop_size = 128
    model_hparams.trim_top_db = 23
    model_hparams.frame_shift_ms = None  # Can replace hop_size parameter. (Recommended: 12.5)
    model_hparams.use_lws = False
    model_hparams.silence_threshold = 2  # silence threshold used for sound trimming for wavenet preprocessing
    model_hparams.trim_silence = True  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    model_hparams.vocoder = 'gl'
    model_hparams.preemphasize = False  # whether to apply filter
    model_hparams.preemphasis = 0.97  # filter coefficient.


class RSQRTSchedule(object):
    def __init__(self, args, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = args.lr 
        self.warmup_updates = args.warmup_updates
        self.hidden_size = args.hidden_size
        self.lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates)**-0.5
        rsqrt_hidden = self.hidden_size**-0.5
        self.lr = constant_lr * warmup * rsqrt_decay * rsqrt_hidden
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class LJSpeechTask(object):

    def __init__(self, args):
        #set_ljspeech_hparams(args)
        self.args = args
        self.ws = getattr(args, 'ws', False)
        self.arch = getattr(args, 'arch', None)
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.max_tokens = args.max_tokens
        self.max_sentences = args.max_sentences
        self.max_eval_tokens = args.max_eval_tokens if getattr(args, 'max_eval_tokens', None) is not None else self.max_tokens
        self.max_eval_sentences = args.max_eval_sentences if getattr(args, 'max_eval_sentences', None) is not None else self.max_sentences
        if isinstance(self.arch, str):
            self.arch = list(map(int, self.arch.strip().split()))
        if self.arch is not None:
            self.num_heads = utils.get_num_heads(self.arch[args.enc_layers:])
        
    def setup_task(self, model_state_dict=None, optimizer_state_dict=None, ws=False):
        self.phone_encoder = self.build_phone_encoder(self.data_dir)
        self.train_dataset, self.valid_dataset, self.test_dataset = self.build_dataset(self.data_dir, self.args.raw_data)
        self.train_queue = self.build_queue(self.train_dataset, True, self.max_tokens, self.max_sentences)
        self.valid_queue = self.build_queue(self.valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
        self.test_queue = self.build_queue(self.test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
        self.model = self.build_model(arch=self.arch, model_state_dict=model_state_dict, ws=self.ws | ws)
        self.optimizer = self.build_optimizer(optimizer_state_dict=optimizer_state_dict)
        self.scheduler = self.build_scheduler()
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        #if torch.cuda.device_count() > 1:
        #    torch.distributed.init_process_group(backend='nccl')

    def setup_search_task(self):
        self.phone_encoder = self.build_phone_encoder(self.data_dir)
        self.train_dataset, self.valid_dataset, self.test_dataset = self.build_dataset(self.data_dir, self.args.raw_data)
        self.train_queue = self.build_queue(self.train_dataset, True, self.max_tokens, self.max_sentences)
        self.valid_queue = self.build_queue(self.valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
        self.test_queue = self.build_queue(self.test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()

    def build_model(self, arch=None, model_state_dict=None, ws=False):
        if arch is None:
            arch = self.arch
        assert (arch is not None) ^ ws, 'arch and ws are mutual'
        if ws:
            _model = model_ws.NASNetwork(self.args, self.phone_encoder)
        else:
            _model = model.NASNetwork(self.args, arch, self.phone_encoder)
        if model_state_dict is not None:
            _model.load_state_dict(model_state_dict)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                _model = nn.DataParallel(_model)
            _model = _model.cuda()
        return _model

    def build_optimizer(self, model=None, optimizer_state_dict=None):
        if model is None:
            model = self.model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            betas=(self.args.optimizer_adam_beta1, self.args.optimizer_adam_beta2),
            weight_decay=self.args.weight_decay)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        return optimizer

    def build_scheduler(self, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        return RSQRTSchedule(self.args, optimizer)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list)

    def split_train_test_set(self, data_dir, test_num=500):
        data_file_name = 'metadata_phone.csv'
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
        total_num = len(data_df.index)
        train_uttids = [i for i in np.arange(0, total_num)]
        test_uttids = []
        for _ in range(test_num):
            random_index = int(np.random.randint(0, len(train_uttids)))
            test_uttids.append(train_uttids[random_index])
            del train_uttids[random_index]

        logging.info(">>test {}".format(len(test_uttids)))
        logging.info(">>train {}".format(len(train_uttids)))
        logging.info(">>total {}".format(len(list(set(test_uttids + train_uttids)))))
        return train_uttids, test_uttids

    def build_dataset(self, data_dir, raw_data):
        if raw_data:
            train_utt_ids, test_utt_ids = self.split_train_test_set(data_dir)
            train_dataset = LJSpeechRawDataset(data_dir, self.args, self.phone_encoder, utt_ids=train_utt_ids, shuffle=True)
            test_dataset = LJSpeechRawDataset(data_dir, self.args, self.phone_encoder, utt_ids=test_utt_ids, shuffle=False)
            valid_dataset = test_dataset
        else:
            train_dataset = LJSpeechDataset(data_dir, 'train', self.args, self.phone_encoder, shuffle=True)
            valid_dataset = LJSpeechDataset(data_dir, 'valid', self.args, self.phone_encoder, shuffle=False)
            test_dataset = LJSpeechDataset(data_dir, 'test', self.args, self.phone_encoder, shuffle=False)
        return train_dataset, valid_dataset, test_dataset

    def build_queue(self, dataset, shuffle, max_tokens=None, max_sentences=None, required_batch_size_multiple=8):
        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= torch.cuda.device_count()
        if max_sentences is not None:
            max_sentences *= torch.cuda.device_count()
        indices = dataset.ordered_indices()
        batch_sampler = utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
        else:
            batches = batch_sampler
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batches, num_workers=8)

    def remove_padding(self, x, hit_eos=None):
        if x is None:
            return None
        if len(x.shape) == 1:  # wav
            hit_eos = np.sum(~hit_eos)
            hit_eos = (hit_eos - 1) * self.args.hop_size + self.args.win_size
            return x[:hit_eos]
        if x.shape[1] > 1:  # mel
            if len(x.shape) == 3:
                x = x[:, :, :1]
            if hit_eos is not None:
                return x[~hit_eos]
            else:
                return x[np.abs(x).sum(1).reshape(-1) != 0]
        else:  # text
            if len(np.where(x <= 1)[0]) > 0:
                x = x[:np.where(x <= 1)[0][0]]
            return x

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def make_stop_target(self, target):
        # target : B x T x mel
        seq_mask = target.abs().sum(-1).ne(0).float()
        seq_length = seq_mask.sum(1)
        mask_r = 1 - utils.sequence_mask(seq_length - 1, target.size(1)).float()
        return seq_mask, mask_r

    def weighted_cross_entropy_with_logits(self, targets, logits, pos_weight=1):
        x = logits
        z = targets
        q = pos_weight
        l = 1 + (q - 1) * z
        return (1 - z) * x + l * (torch.log(1 + torch.exp(-x.abs())) + F.relu(-x))

    def loss(self, decoder_output, target):
        # decoder_output : B x T x (mel+1)
        # target : B x T x mel
        if target is None:
            return {
                'decoder loss': decoder_output.new(1).fill_(0)[0],
                'stop loss': decoder_output.new(1).fill_(0)[0],
            }

        predicted_mel = decoder_output[:, :, :self.args.audio_num_mel_bins]
        predicted_stop = decoder_output[:, :, -1]
        seq_mask, stop_mask = self.make_stop_target(target)

        decoder_loss = F.mse_loss(predicted_mel, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        decoder_loss = (decoder_loss * weights).sum() / weights.sum()
        stop_loss = (self.weighted_cross_entropy_with_logits(stop_mask, predicted_stop, self.args.stop_token_weight) * seq_mask).sum()
        stop_loss = stop_loss / (seq_mask.sum() + target.size(0) * (self.args.stop_token_weight - 1))
        
        return {
            'decoder loss': decoder_loss,
            'stop loss': stop_loss,
        }

    def train(self, model=None, optimizer=None, scheduler=None, epoch=1, num_updates=0, arch_pool=None, arch_pool_prob=None):
        if model is None:
            model = self.model
        if optimizer is None:
            optimizer = self.optimizer
        if scheduler is None:
            scheduler = self.scheduler
        
        decoder_loss_obj = utils.AvgrageMeter()
        stop_loss_obj = utils.AvgrageMeter()
        loss_obj = utils.AvgrageMeter()
        model.train()
        for step, sample in enumerate(self.train_queue):
            num_updates += 1
            scheduler.step(num_updates)
            input = utils.move_to_cuda(sample['src_tokens'])
            prev_output_mels = utils.move_to_cuda(sample['prev_output_mels'])
            target = utils.move_to_cuda(sample['targets'])
            optimizer.zero_grad()

            if arch_pool is not None:# ws train
                arch = utils.sample_arch(arch_pool, arch_pool_prob)
                output, _ = model(input, prev_output_mels, target, arch)
            else:# normal train
                output, _ = model(input, prev_output_mels, target)
            loss_output = self.loss(output, target)
            decoder_loss = loss_output['decoder loss']
            stop_loss = loss_output['stop loss']
            total_loss = decoder_loss + stop_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
            optimizer.step()

            n = sample['nmels']
            decoder_loss_obj.update(decoder_loss.data, n)
            stop_loss_obj.update(stop_loss.data, n)
            loss_obj.update(total_loss.data, n)

            if (step+1) % self.args.log_interval == 0:
                lr = scheduler.get_lr()
                log_output = 'train {}@{} global step {} lr {:.6f} decoder loss {:.6f} stop loss {:.6f} total loss {:.6f}'.format(
                        epoch, step+1, num_updates, lr, decoder_loss_obj.avg, stop_loss_obj.avg, loss_obj.avg)
                if arch_pool is not None:
                    log_output = 'arch {}\n'.format(arch) + log_output
                logging.info(log_output)

        return decoder_loss_obj.avg, stop_loss_obj.avg, loss_obj.avg, num_updates

    def valid(self, model=None):
        if model is None:
            model = self.model

        decoder_loss_obj = utils.AvgrageMeter()
        stop_loss_obj = utils.AvgrageMeter()
        loss_obj = utils.AvgrageMeter()
        fr_obj = utils.AvgrageMeter()
        pcr_obj = utils.AvgrageMeter()
        dfr_obj = utils.AvgrageMeter()
        model.eval()
        with torch.no_grad():
            for step, sample in enumerate(self.test_queue):
                input = utils.move_to_cuda(sample['src_tokens'])
                prev_output_mels = utils.move_to_cuda(sample['prev_output_mels'])
                target = utils.move_to_cuda(sample['targets'])
                if target is not None:
                    output, attn_logits = model(input, prev_output_mels, target)
                else:
                    bsz = input.size(0)
                    max_input_len = input.size(1)
                    if type(model) is nn.DataParallel:
                        model = model.module
                    decode_length = self.estimate_decode_length(max_input_len)
                    encoder_outputs = model.forward_encoder(input)
                    encoder_out = encoder_outputs['encoder_out']
                    encoder_padding_mask = encoder_outputs['encoder_padding_mask']
                    decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
                    output = input.new(bsz, 0, self.args.audio_num_mel_bins+1).fill_(self.padding_idx).float()
                    hit_eos = input.new(bsz, 1).fill_(0).bool()
                    stop_logits = input.new(bsz, 0).fill_(0).float()
                    encdec_attn_logits = []
                    num_heads = self.num_heads
                    for i in range(self.args.dec_layers):
                        encdec_attn_logits.append(input.new(bsz, num_heads[i], 0, max_input_len).fill_(0).float())
                    incremental_state = {}
                    for step in range(decode_length):
                        decoder_output, attn_logits = model.forward_decoder(decoder_input[:, :step+1], encoder_out, encoder_padding_mask, incremental_state=incremental_state)
                        next_mel = decoder_output[:, -1:, :self.args.audio_num_mel_bins]
                        stop_logit = decoder_output[:, -1:, -1]
                        stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
                        output = torch.cat((output, decoder_output), dim=1)
                        for i in range(self.args.dec_layers):
                            encdec_attn_logits[i] = torch.cat((encdec_attn_logits[i], attn_logits[i]), dim=2)
                        decoder_input[:, step+1] = next_mel[:, -1]
                    attn_logits = encdec_attn_logits
                    this_hit_eos = hit_eos[:, -1:]
                    this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
                    hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)


                loss_output = self.loss(output, target)
                decoder_loss = loss_output['decoder loss']
                stop_loss = loss_output['stop loss']
                total_loss = decoder_loss + stop_loss
                
                n = sample['nmels'] if sample['nmels'] is not None else sample['nsamples']
                decoder_loss_obj.update(decoder_loss.data, n)
                stop_loss_obj.update(stop_loss.data, n)
                loss_obj.update(total_loss.data, n)
                
                encdec_attn = utils.select_attn(attn_logits)
                
                src_lengths = utils.move_to_cuda(sample['src_lengths']) #- 1 # exclude eos
                if target is not None:
                    target_lengths = utils.move_to_cuda(sample['target_lengths'])
                    target_padding_mask = target.abs().sum(-1).eq(self.padding_idx)
                else:
                    hit_eos = hit_eos[:, 1:]
                    target_lengths = (1.0 - hit_eos.float()).sum(dim=-1)
                    target_padding_mask = output[:, :, :self.args.audio_num_mel_bins].abs().sum(-1).eq(self.padding_idx)
                src_padding_mask = input.eq(self.padding_idx)# | input.eq(self.eos_idx)  # also exclude eos
                src_seg_mask = input.eq(self.seg_idx)
                focus_rate = utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask).mean()
                phone_coverage_rate = utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask, target_padding_mask).mean()
                attn_ks = src_lengths.float() / target_lengths.float()
                diagonal_focus_rate = utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths, src_padding_mask, target_padding_mask).mean()

                fr_obj.update(focus_rate.data, sample['nsamples'])
                pcr_obj.update(phone_coverage_rate.data, sample['nsamples'])
                dfr_obj.update(diagonal_focus_rate.data, sample['nsamples'])

        return decoder_loss_obj.avg, stop_loss_obj.avg, loss_obj.avg, fr_obj.avg, pcr_obj.avg, dfr_obj.avg

    def infer_batch(self, model, sample):
        if model is None:
            model = self.model
        
        model.eval()
        if type(model) is nn.DataParallel:
            model = model.module
        with torch.no_grad():
            input = utils.move_to_cuda(sample['src_tokens'])
            prev_output_mels = utils.move_to_cuda(sample['prev_output_mels'])
            target = utils.move_to_cuda(sample['targets'])
            bsz = input.size(0)
            max_input_len = input.size(1)

            if self.args.gta:
                decode_length = target.size(1)
            else:
                decode_length = self.estimate_decode_length(max_input_len)
            
            encoder_outputs = model.forward_encoder(input)
            encoder_out = encoder_outputs['encoder_out']
            encoder_padding_mask = encoder_outputs['encoder_padding_mask']

            hit_eos = input.new(bsz, 1).fill_(0).bool()
            stop_logits = input.new(bsz, 0).fill_(0).float()
            stage = 0
            decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
            decoded_mel = input.new(bsz, 0, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
            encdec_attn_logits = []

            for i in range(self.args.dec_layers):
                encdec_attn_logits.append(input.new(bsz, self.num_heads[i], 0, max_input_len).fill_(0).float())
            #encdec_attn_logits = input.new(bsz, self.args.dec_layers, 0, max_input_len).fill_(0).float()
            attn_pos = input.new(bsz).fill_(0).int()
            use_masks = []
            for i in range(self.args.dec_layers):
                use_masks.append(input.new(self.num_heads[i]).fill_(0).float())
            #use_masks = input.new(self.args.dec_layers*2).fill_(0).float()
            
            incremental_state = {}
            step = 0
            if self.args.attn_constraint:
                for i, layer in enumerate(model.decoder.layers):
                    enc_dec_attn_constraint_mask = input.new(bsz, self.num_heads[i], max_input_len).fill_(0).int()
                    layer.set_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)
            while True:
                #import pdb; pdb.set_trace()
                if self.is_finished(step, decode_length, hit_eos, stage):
                    break
                
                if self.args.gta:
                    decoder_input[:, step] = prev_output_mels[:, step]

                decoder_output, attn_logits = model.forward_decoder(decoder_input[:, :step+1], encoder_out, encoder_padding_mask, incremental_state=incremental_state)
                next_mel = decoder_output[:, -1:, :self.args.audio_num_mel_bins]
                stop_logit = decoder_output[:, -1:, -1]
                stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
                decoded_mel = torch.cat((decoded_mel, next_mel), dim=1)
                for i in range(self.args.dec_layers):
                    encdec_attn_logits[i] = torch.cat((encdec_attn_logits[i], attn_logits[i]), dim=2)
                step += 1

                this_hit_eos = hit_eos[:, -1:]
                if self.args.attn_constraint:
                    this_hit_eos |= (attn_pos[:, None] >= (encoder_padding_mask < 1.0).float().sum(dim=-1, keepdim=True).int() - 5) & (torch.sigmoid(stop_logit) > 0.5)
                else:
                    this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
                hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)

                
                if not self.args.gta:
                    decoder_input[:, step] = next_mel[:, -1]

                if self.args.attn_constraint:
                    stage_change_step = 50
                    all_prev_weights = []
                    for i in range(self.args.dec_layers):
                        all_prev_weights.append(torch.softmax(encdec_attn_logits[i], dim=-1)) # bsz x head x L x L_kv

                    # if the stage should change
                    next_stage = (step == stage_change_step) | (step >= decode_length)
                    if not self.args.gta:
                        next_stage |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
                    next_stage &= (stage == 0)

                    # choose the diagonal attention
                    if next_stage:#TODO
                        use_masks = []
                        for i in range(self.args.dec_layers):
                            use_mask = (all_prev_weights[i][:, :, :step].max(dim=-1).values.mean(dim=(0, 2)) > 0.6).float() # [head]
                            use_masks.append(use_mask)
                        attn_pos = input.new(bsz).fill_(0).int()

                        # reseet when the stage changes
                        for layer in model.decoder.layers:
                            layer.clear_buffer(input, encoder_out, encoder_padding_mask, incremental_state)
                        
                        encdec_attn_logits = []
                        for i in range(self.args.dec_layers):
                            encdec_attn_logits.append(input.new(bsz, self.num_heads[i], 0, max_input_len).fill_(0).float())
                        decoded_mel = input.new(bsz, 0, self.args.audio_num_mel_bins).fill_(0).float()
                        decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(0).float()
                        hit_eos = input.new(bsz, 1).fill_(0).bool()
                        stage = stage + 1
                        step = 0

                    prev_weights_mask1 = utils.sequence_mask(torch.max(attn_pos - 1, attn_pos.new(attn_pos.size()).fill_(0)).float(), encdec_attn_logits[0].size(-1)).float() # bsz x L_kv
                    prev_weights_mask2 = 1.0 - utils.sequence_mask(attn_pos.float() + 4, encdec_attn_logits[0].size(-1)).float()   # bsz x L_kv
                    enc_dec_attn_constraint_masks = []
                    for i in range(self.args.dec_layers):
                        mask = (prev_weights_mask1 + prev_weights_mask2)[:, None, :] * use_masks[i][None, :, None] # bsz x head x L_kv
                        enc_dec_attn_constraint_masks.append(mask)
                    #enc_dec_attn_constraint_masks = (prev_weights_mask1 + prev_weights_mask2)[:, None, None, :] * use_masks[None, :, None, None] # bsz x (n_layers x head) x 1 x L_kv

                    for i, layer in enumerate(model.decoder.layers):
                        enc_dec_attn_constraint_mask = enc_dec_attn_constraint_masks[i]
                        layer.set_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)

                    def should_move_on():
                        prev_weights = []
                        for i in range(self.args.dec_layers):
                            prev_weight = (all_prev_weights[i] * use_masks[i][None, :, None, None]).sum(dim=1)
                            prev_weights.append(prev_weight)
                        prev_weights = sum(prev_weights) / sum([mask.sum() for mask in use_masks])
                        #prev_weights = (prev_weights * use_masks[None, :, None, None]).sum(dim=1) / use_masks.sum()
                        move_on = (prev_weights[:, -3:].mean(dim=1).gather(1, attn_pos[:, None].long())).squeeze() < 0.7
                        move_on &= torch.argmax(prev_weights[:, -1], -1) > attn_pos.long()
                        return attn_pos + move_on.int()
                        
                    if step > 3 and stage == 1:
                        attn_pos = should_move_on()

            #size = encdec_attn_logits.size()
            #encdec_attn_logits = encdec_attn_logits.view(size[0], size[1]*size[2], size[3], size[4])
            encdec_attn = utils.select_attn(encdec_attn_logits)

            src_lengths = utils.move_to_cuda(sample['src_lengths']) - 1 # exclude eos
            target_lengths = (1.0 - hit_eos[:, 1:].float()).sum(dim=-1) + 1
            src_padding_mask = input.eq(self.padding_idx) | input.eq(self.eos_idx)  # also exclude eos
            src_seg_mask = input.eq(self.seg_idx)
            target_padding_mask = decoded_mel.abs().sum(-1).eq(self.padding_idx)
            focus_rate = utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
            phone_coverage_rate = utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask, target_padding_mask)
            attn_ks = src_lengths.float() / target_lengths.float()
            diagonal_focus_rate = utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths, src_padding_mask, target_padding_mask)

            return decoded_mel, encdec_attn.unsqueeze(1), hit_eos, stop_logits, focus_rate, phone_coverage_rate, diagonal_focus_rate

    def is_finished(self, step, decode_length, hit_eos, stage):
        finished = step >= decode_length
        if not self.args.gta:
            finished |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
        if self.args.attn_constraint:
            finished &= stage != 0
        return finished

    def infer(self, model=None, split='test'):
        if model is None:
            model = self.model
        if split == 'train':
            queue = self.train_queue
        elif split == 'valid':
            queue = self.valid_queue
        else:
            assert split == 'test'
            queue = self.test_queue

        nsamples_finished = 0
        for batch, sample in enumerate(queue):
            logging.info('inferring batch {} with {} samples'.format(batch+1, sample['nsamples']))
            decoded_mel, encdec_attn, hit_eos, _, focus_rate, phone_coverage_rate, diagonal_focus_rate = self.infer_batch(model, sample)

            hit_eos = hit_eos[:, 1:]
            outputs = decoded_mel
            predict_lengths = (1.0 - hit_eos.float()).sum(dim=-1)
            outputs *= (1.0 - hit_eos.float())[:, :, None]

            sample['outputs'] = outputs
            sample['predict_mels'] = decoded_mel
            sample['predict_lengths'] = predict_lengths
            sample['encdec_attn'] = encdec_attn
            sample['focus_rate'] = focus_rate
            sample['phone_coverage_rate'] = phone_coverage_rate
            sample['diagonal_focus_rate'] = diagonal_focus_rate
            self.after_infer(sample)
            nsamples_finished += sample['nsamples']
            
    def valid_for_search(self, model=None, gta=False, arch_pool=None, layer_norm_training=False):
        if model is None:
            model = self.model
        if arch_pool is not None:
            loss_res, fr_res, pcr_res, dfr_res = [], [], [] ,[]
            for arch in arch_pool:
                loss_obj = utils.AvgrageMeter()
                fr_obj = utils.AvgrageMeter()
                pcr_obj = utils.AvgrageMeter()
                dfr_obj = utils.AvgrageMeter()
                for batch, sample in enumerate(self.valid_queue):
                    ret = self.valid_for_search_batch(model, sample, gta, arch, layer_norm_training)
                    loss_obj.update(ret['loss'], ret['nsamples'])
                    fr_obj.update(ret['focus_rate'], ret['nsamples'])
                    pcr_obj.update(ret['phone_coverage_rate'], ret['nsamples'])
                    dfr_obj.update(ret['diagonal_focus_rate'], ret['nsamples'])
                loss_res.append(loss_obj.avg)
                fr_res.append(fr_obj.avg)
                pcr_res.append(pcr_obj.avg)
                dfr_res.append(dfr_obj.avg)
            return fr_res, pcr_res, dfr_res, loss_res
        
        loss_obj = utils.AvgrageMeter()
        fr_obj = utils.AvgrageMeter()
        pcr_obj = utils.AvgrageMeter()
        dfr_obj = utils.AvgrageMeter()
        for batch, sample in enumerate(self.valid_queue):
            ret = self.valid_for_search_batch(model, sample, gta)
            loss_obj.update(ret['loss'], ret['nsamples'])
            fr_obj.update(ret['focus_rate'], ret['nsamples'])
            pcr_obj.update(ret['phone_coverage_rate'], ret['nsamples'])
            dfr_obj.update(ret['diagonal_focus_rate'], ret['nsamples'])
        return fr_obj.avg, pcr_obj.avg, dfr_obj.avg, loss_obj.avg

    def valid_for_search_batch(self, model, sample, gta=False, arch=None, layer_norm_training=False):
        if model is None:
            model = self.model
        model.eval()

        with torch.no_grad():
            input = utils.move_to_cuda(sample['src_tokens'])
            prev_output_mels = utils.move_to_cuda(sample['prev_output_mels'])
            target = utils.move_to_cuda(sample['targets'])
            bsz = input.size(0)
            max_input_len = input.size(1)

            if gta:
                output, attn_logits = model(input, prev_output_mels, target, arch=arch, layer_norm_training=layer_norm_training)
                encdec_attn_logits = attn_logits
            else:
                if type(model) is nn.DataParallel:
                    model = model.module
                decode_length = target.size(1) if target is not None else self.estimate_decode_length(input.size(1))
                encoder_outputs = model.forward_encoder(input, arch=arch, layer_norm_training=layer_norm_training)
                encoder_out = encoder_outputs['encoder_out']
                encoder_padding_mask = encoder_outputs['encoder_padding_mask']
                decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
                output = input.new(bsz, 0, self.args.audio_num_mel_bins+1).fill_(self.padding_idx).float()
                hit_eos = input.new(bsz, 1).fill_(0).bool()
                stop_logits = input.new(bsz, 0).fill_(0).float()
                encdec_attn_logits = []
                if arch is not None: # in ws mode, arch is provided at run
                    num_heads = utils.get_num_heads(arch[self.args.enc_layers:])
                else: # in general mode, arch is defined at the begining
                    num_heads = self.num_heads
                for i in range(self.args.dec_layers):
                    encdec_attn_logits.append(input.new(bsz, num_heads[i], 0, max_input_len).fill_(0).float())
                incremental_state = {}
                for step in range(decode_length):
                    decoder_output, attn_logits = model.forward_decoder(decoder_input[:, :step+1], encoder_out, encoder_padding_mask, incremental_state=incremental_state, arch=arch, layer_norm_training=layer_norm_training)
                    next_mel = decoder_output[:, -1:, :self.args.audio_num_mel_bins]
                    stop_logit = decoder_output[:, -1:, -1]
                    stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
                    output = torch.cat((output, decoder_output), dim=1)
                    for i in range(self.args.dec_layers):
                        encdec_attn_logits[i] = torch.cat((encdec_attn_logits[i], attn_logits[i]), dim=2)
                    decoder_input[:, step+1] = next_mel[:, -1]
                    this_hit_eos = hit_eos[:, -1:]
                    this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
                    hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)
            
            loss_output = self.loss(output, target)
            decoder_loss = loss_output['decoder loss']
            stop_loss = loss_output['stop loss']
            total_loss = decoder_loss + stop_loss
            encdec_attn = utils.select_attn(encdec_attn_logits)

            src_lengths = utils.move_to_cuda(sample['src_lengths']) - 1 # exclude eos
            if target is not None:
                target_lengths = utils.move_to_cuda(sample['target_lengths'])
                target_padding_mask = target.abs().sum(-1).eq(self.padding_idx)
            else:
                hit_eos = hit_eos[:, 1:]
                target_lengths = (1.0 - hit_eos.float()).sum(dim=-1)
                target_padding_mask = output[:, :, :self.args.audio_num_mel_bins].abs().sum(-1).eq(self.padding_idx)
            src_padding_mask = input.eq(self.padding_idx) | input.eq(self.eos_idx)  # also exclude eos
            src_seg_mask = input.eq(self.seg_idx)
            focus_rate = utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
            phone_coverage_rate = utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask, target_padding_mask)
            attn_ks = src_lengths.float() / target_lengths.float()
            diagonal_focus_rate = utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths, src_padding_mask, target_padding_mask)

            ret = {
                'focus_rate': focus_rate.mean().data,
                'phone_coverage_rate': phone_coverage_rate.mean().data,
                'diagonal_focus_rate': diagonal_focus_rate.mean().data,
                'loss': total_loss.data,
                'nsamples': sample['nsamples'],
            }
            return ret

    def estimate_decode_length(self, input_length):
        return input_length * 5 + 100

    def after_infer(self, predictions):
        predictions = utils.unpack_dict_to_list(predictions)
        for num_predictions, prediction in enumerate(predictions):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id', None)
            text = prediction.get('text', None)
            src_tokens = prediction.get('src_tokens')
            src_lengths = prediction.get('src_lengths')
            targets = prediction.get("targets", None)
            outputs = prediction["outputs"]
            focus_rate = prediction['focus_rate']
            phone_coverage_rate = prediction['phone_coverage_rate']
            diagonal_focus_rate = prediction['diagonal_focus_rate']
            decoded_inputs_txt = self.phone_encoder.decode(src_tokens, strip_eos=True, strip_padding=True)
            out_wav = audio.inv_mel_spectrogram(outputs.T, self.args)
            prediction['out_wav'] = out_wav

            if prediction.get('hit_eos') is None:
                assert prediction.get('predict_lengths') is not None
                prediction['hit_eos'] = np.arange(outputs.shape[0]) >= prediction['predict_lengths']
            
            targets = self.remove_padding(targets) if targets is not None else None # speech
            outputs = self.remove_padding(outputs, prediction.get('hit_eos'))
            if out_wav is not None:
                outputs = self.remove_padding(out_wav, prediction.get('hit_eos'))

            prediction['predict_mels'] = self.remove_padding(prediction['predict_mels'], prediction.get('hit_eos'))

            if 'encdec_attn' in prediction:
                encdec_attn = prediction['encdec_attn']
                encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
                prediction['encdec_attn'] = self.remove_padding(encdec_attn, prediction.get('hit_eos'))
                prediction['encdec_attn'] = prediction['encdec_attn'].T[:src_lengths]

            if hasattr(self.args, 'checkpoint_path') and self.args.checkpoint_path is not None:
                steps = os.path.split(self.args.checkpoint_path)[-1].strip().split('checkpoint')[1].split('.pt')[0]
                output_dir = os.path.join(self.args.output_dir, f'generated_{steps}')
            else:
                output_dir = os.path.join(self.args.output_dir, f'generated')
            os.makedirs(output_dir, exist_ok=True)

            def log_audio(outputs, prefix, alignment=None, mels=None, decoded_txt=None):
                if len(outputs.shape) == 1:
                    wav_out = outputs
                else:
                    mel = outputs.reshape(-1, self.args.audio_num_mel_bins)
                    wav_out = audio.inv_mel_spectrogram(mel.T, self.args)
                wav_out = audio.inv_preemphasis(wav_out, self.args.preemphasis, self.args.preemphasize)
                #audio.save_wav(wav_out, os.path.join(output_dir, '[W][{}][{}]{}.wav'.format(prefix, utt_id, text.replace(':', '%3A') if text is not None else '')),
                #           self.args.audio_sample_rate)
                audio.save_wav(wav_out, os.path.join(output_dir, '[W][{}][{}].wav'.format(prefix, utt_id)),
                            self.args.audio_sample_rate)
                #audio.plot_spec(mels.reshape(-1, 80).T,
                #                os.path.join(output_dir, '[P][{}][{}]{}.png'.format(prefix, utt_id, text.replace(':', '%3A') if text is not None else '')))
                audio.plot_spec(mels.reshape(-1, 80).T,
                                os.path.join(output_dir, '[P][{}][{}].png'.format(prefix, utt_id)))
                if alignment is not None:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 16))
                    im = ax.imshow(alignment, aspect='auto', origin='lower',
                                    interpolation='none')
                    decoded_txt = decoded_txt.split(" ")
                    ax.set_yticks(np.arange(len(decoded_txt)))
                    ax.set_yticklabels(list(decoded_txt), fontsize=6)

                    fig.colorbar(im, ax=ax)
                    #fig.savefig(os.path.join(output_dir, '[A][{}][{}]{}.png'.format(prefix, utt_id, text.replace(':', '%3A') if text is not None else '')), format='png')
                    fig.savefig(os.path.join(output_dir, '[A][{}][{}].png'.format(prefix, utt_id)), format='png')
                    plt.close()

                    #with open(os.path.join(output_dir, '[A][{}][{}]{}.txt'.format(prefix, utt_id, text.replace(':', '%3A') if text is not None else '')), 'w') as f:
                    with open(os.path.join(output_dir, '[A][{}][{}].txt'.format(prefix, utt_id)), 'w') as f:
                        f.write('fr %.6f pcr %.6f dfr %.6f' % (focus_rate, phone_coverage_rate, diagonal_focus_rate))

            log_audio(outputs, 'pred', prediction.get('encdec_attn', None), prediction["predict_mels"], decoded_inputs_txt)
            logging.info('pred_outputs.shape: {}'.format(prediction['predict_mels'].shape))
            if targets is not None:
                log_audio(targets, 'gt', None, targets[:, :self.args.audio_num_mel_bins], decoded_inputs_txt)

        logging.info(">>> {}".format(num_predictions+1))


class LJSpeechTaskMB(LJSpeechTask):

    def __init__(self, args):
        self.args = args
        self.ws = getattr(args, 'ws', False)
        self.arch = getattr(args, 'arch', None)
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.max_tokens = args.max_tokens
        self.max_sentences = args.max_sentences
        self.max_eval_tokens = args.max_eval_tokens if getattr(args, 'max_eval_tokens', None) is not None else self.max_tokens
        self.max_eval_sentences = args.max_eval_sentences if getattr(args, 'max_eval_sentences', None) is not None else self.max_sentences
        if isinstance(self.arch, str):
            self.arch = list(map(int, self.arch.strip().split()))
        if self.arch is not None:
            self.num_heads = utils.get_num_heads(self.arch[args.enc_layers*2:])
        
    def setup_task(self, model_state_dict=None, optimizer_state_dict=None, ws=False):
        self.phone_encoder = self.build_phone_encoder(self.data_dir)
        self.train_dataset, self.valid_dataset, self.test_dataset = self.build_dataset(self.data_dir, self.args.raw_data)
        self.train_queue = self.build_queue(self.train_dataset, True, self.max_tokens, self.max_sentences)
        self.valid_queue = self.build_queue(self.valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
        self.test_queue = self.build_queue(self.test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
        self.model = self.build_model(arch=self.arch, model_state_dict=model_state_dict, ws=self.ws | ws)
        self.optimizer = self.build_optimizer(optimizer_state_dict=optimizer_state_dict)
        self.scheduler = self.build_scheduler()
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        #if torch.cuda.device_count() > 1:
        #    torch.distributed.init_process_group(backend='nccl')

    def build_model(self, arch=None, model_state_dict=None, ws=False):
        if arch is None:
            arch = self.arch
        assert (arch is not None) ^ ws, 'arch and ws are mutual'
        if ws:
            _model = model_ws.NASNetwork(self.args, self.phone_encoder)
        else:
            _model = model.NASNetwork(self.args, arch, self.phone_encoder)
        if model_state_dict is not None:
            _model.load_state_dict(model_state_dict)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                _model = nn.DataParallel(_model)
            _model = _model.cuda()
        return _model

    def infer_batch(self, model, sample):
        if model is None:
            model = self.model
        
        model.eval()
        if type(model) is nn.DataParallel:
            model = model.module
        with torch.no_grad():
            input = utils.move_to_cuda(sample['src_tokens'])
            prev_output_mels = utils.move_to_cuda(sample['prev_output_mels'])
            target = utils.move_to_cuda(sample['targets'])
            bsz = input.size(0)
            max_input_len = input.size(1)

            if self.args.gta:
                decode_length = target.size(1)
            else:
                decode_length = self.estimate_decode_length(max_input_len)
            
            encoder_outputs = model.forward_encoder(input)
            encoder_out = encoder_outputs['encoder_out']
            encoder_padding_mask = encoder_outputs['encoder_padding_mask']

            hit_eos = input.new(bsz, 1).fill_(0).bool()
            stop_logits = input.new(bsz, 0).fill_(0).float()
            stage = 0
            decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
            decoded_mel = input.new(bsz, 0, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
            encdec_attn_logits = []

            for i in range(self.args.dec_layers):
                encdec_attn_logits.append(input.new(bsz, self.num_heads[2*i], 0, max_input_len).fill_(0).float())
                encdec_attn_logits.append(input.new(bsz, self.num_heads[2*i+1], 0, max_input_len).fill_(0).float())
            #encdec_attn_logits = input.new(bsz, self.args.dec_layers, 0, max_input_len).fill_(0).float()
            attn_pos = input.new(bsz).fill_(0).int()
            use_masks = []
            for i in range(self.args.dec_layers):
                use_masks.append(input.new(self.num_heads[2*i]).fill_(0).float())
                use_masks.append(input.new(self.num_heads[2*i+1]).fill_(0).float())
            #use_masks = input.new(self.args.dec_layers*2).fill_(0).float()
            
            incremental_state = {}
            step = 0
            if self.args.attn_constraint:
                for i, layer in enumerate(model.decoder.layers):
                    enc_dec_attn_constraint_mask = input.new(bsz, self.num_heads[2*i], max_input_len).fill_(0).int()
                    layer.set_left_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)
                    enc_dec_attn_constraint_mask = input.new(bsz, self.num_heads[2*i+1], max_input_len).fill_(0).int()
                    layer.set_right_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)
            while True:
                #import pdb; pdb.set_trace()
                if self.is_finished(step, decode_length, hit_eos, stage):
                    break
                
                if self.args.gta:
                    decoder_input[:, step] = prev_output_mels[:, step]

                decoder_output, attn_logits = model.forward_decoder(decoder_input[:, :step+1], encoder_out, encoder_padding_mask, incremental_state=incremental_state)
                next_mel = decoder_output[:, -1:, :self.args.audio_num_mel_bins]
                stop_logit = decoder_output[:, -1:, -1]
                stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
                decoded_mel = torch.cat((decoded_mel, next_mel), dim=1)
                for i in range(self.args.dec_layers):
                    encdec_attn_logits[2*i] = torch.cat((encdec_attn_logits[2*i], attn_logits[2*i]), dim=2)
                    encdec_attn_logits[2*i+1] = torch.cat((encdec_attn_logits[2*i+1], attn_logits[2*i+1]), dim=2)
                step += 1

                this_hit_eos = hit_eos[:, -1:]
                if self.args.attn_constraint:
                    this_hit_eos |= (attn_pos[:, None] >= (encoder_padding_mask < 1.0).float().sum(dim=-1, keepdim=True).int() - 5) & (torch.sigmoid(stop_logit) > 0.5)
                else:
                    this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
                hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)

                
                if not self.args.gta:
                    decoder_input[:, step] = next_mel[:, -1]

                if self.args.attn_constraint:
                    stage_change_step = 50
                    all_prev_weights = []
                    for i in range(self.args.dec_layers):
                        all_prev_weights.append(torch.softmax(encdec_attn_logits[2*i], dim=-1)) # bsz x head x L x L_kv
                        all_prev_weights.append(torch.softmax(encdec_attn_logits[2*i+1], dim=-1))

                    # if the stage should change
                    next_stage = (step == stage_change_step) | (step >= decode_length)
                    if not self.args.gta:
                        next_stage |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
                    next_stage &= (stage == 0)

                    # choose the diagonal attention
                    if next_stage:#TODO
                        use_masks = []
                        for i in range(self.args.dec_layers):
                            use_mask = (all_prev_weights[2*i][:, :, :step].max(dim=-1).values.mean(dim=(0, 2)) > 0.6).float() # [head]
                            use_masks.append(use_mask)
                            use_mask = (all_prev_weights[2*i+1][:, :, :step].max(dim=-1).values.mean(dim=(0, 2)) > 0.6).float() # [head]
                            use_masks.append(use_mask)
                        attn_pos = input.new(bsz).fill_(0).int()

                        # reset when the stage changes
                        for i, layer in enumerate(model.decoder.layers):
                            layer.clear_left_buffer(input, encoder_out, encoder_padding_mask, incremental_state)
                            layer.clear_right_buffer(input, encoder_out, encoder_padding_mask, incremental_state)
                        
                        encdec_attn_logits = []
                        for i in range(self.args.dec_layers):
                            encdec_attn_logits.append(input.new(bsz, self.num_heads[2*i], 0, max_input_len).fill_(0).float())
                            encdec_attn_logits.append(input.new(bsz, self.num_heads[2*i+1], 0, max_input_len).fill_(0).float())
                        decoded_mel = input.new(bsz, 0, self.args.audio_num_mel_bins).fill_(0).float()
                        decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(0).float()
                        hit_eos = input.new(bsz, 1).fill_(0).bool()
                        stage = stage + 1
                        step = 0

                    prev_weights_mask1 = utils.sequence_mask(torch.max(attn_pos - 1, attn_pos.new(attn_pos.size()).fill_(0)).float(), encdec_attn_logits[0].size(-1)).float() # bsz x L_kv
                    prev_weights_mask2 = 1.0 - utils.sequence_mask(attn_pos.float() + 4, encdec_attn_logits[0].size(-1)).float()   # bsz x L_kv
                    enc_dec_attn_constraint_masks = []
                    for i in range(self.args.dec_layers):
                        mask = (prev_weights_mask1 + prev_weights_mask2)[:, None, :] * use_masks[2*i][None, :, None] # bsz x head x L_kv
                        enc_dec_attn_constraint_masks.append(mask)
                        mask = (prev_weights_mask1 + prev_weights_mask2)[:, None, :] * use_masks[2*i+1][None, :, None] # bsz x head x L_kv
                        enc_dec_attn_constraint_masks.append(mask)
                    #enc_dec_attn_constraint_masks = (prev_weights_mask1 + prev_weights_mask2)[:, None, None, :] * use_masks[None, :, None, None] # bsz x (n_layers x head) x 1 x L_kv

                    for i, layer in enumerate(model.decoder.layers):
                        enc_dec_attn_constraint_mask = enc_dec_attn_constraint_masks[2*i]
                        layer.set_left_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)
                        enc_dec_attn_constraint_mask = enc_dec_attn_constraint_masks[2*i+1]
                        layer.set_right_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)

                    def should_move_on():
                        prev_weights = []
                        for i in range(self.args.dec_layers):
                            prev_weight = (all_prev_weights[2*i] * use_masks[2*i][None, :, None, None]).sum(dim=1)
                            prev_weights.append(prev_weight)
                            prev_weight = (all_prev_weights[2*i+1] * use_masks[2*i+1][None, :, None, None]).sum(dim=1)
                            prev_weights.append(prev_weight)
                        prev_weights = sum(prev_weights) / sum([mask.sum() for mask in use_masks])
                        #prev_weights = (prev_weights * use_masks[None, :, None, None]).sum(dim=1) / use_masks.sum()
                        move_on = (prev_weights[:, -3:].mean(dim=1).gather(1, attn_pos[:, None].long())).squeeze() < 0.7
                        move_on &= torch.argmax(prev_weights[:, -1], -1) > attn_pos.long()
                        return attn_pos + move_on.int()
                        
                    if step > 3 and stage == 1:
                        attn_pos = should_move_on()

            #size = encdec_attn_logits.size()
            #encdec_attn_logits = encdec_attn_logits.view(size[0], size[1]*size[2], size[3], size[4])
            encdec_attn = utils.select_attn(encdec_attn_logits)

            src_lengths = utils.move_to_cuda(sample['src_lengths']) - 1 # exclude eos
            target_lengths = (1.0 - hit_eos[:, 1:].float()).sum(dim=-1) + 1
            src_padding_mask = input.eq(self.padding_idx) | input.eq(self.eos_idx)  # also exclude eos
            src_seg_mask = input.eq(self.seg_idx)
            target_padding_mask = decoded_mel.abs().sum(-1).eq(self.padding_idx)
            focus_rate = utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
            phone_coverage_rate = utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask, target_padding_mask)
            attn_ks = src_lengths.float() / target_lengths.float()
            diagonal_focus_rate = utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths, src_padding_mask, target_padding_mask)

            return decoded_mel, encdec_attn.unsqueeze(1), hit_eos, stop_logits, focus_rate, phone_coverage_rate, diagonal_focus_rate
            
    def valid_for_search_batch(self, model, sample, gta=False, arch=None, layer_norm_training=False):
        if model is None:
            model = self.model
        model.eval()
        if type(model) is nn.DataParallel:
            model = model.module

        with torch.no_grad():
            input = utils.move_to_cuda(sample['src_tokens'])
            prev_output_mels = utils.move_to_cuda(sample['prev_output_mels'])
            target = utils.move_to_cuda(sample['targets'])
            bsz = input.size(0)
            max_input_len = input.size(1)

            if gta:
                output, attn_logits = model(input, prev_output_mels, target, arch=arch, layer_norm_training=layer_norm_training)
                encdec_attn_logits = attn_logits
            else:
                decode_length = target.size(1) if target is not None else self.estimate_decode_length(input.size(1))
                encoder_outputs = model.forward_encoder(input, arch=arch, layer_norm_training=layer_norm_training)
                encoder_out = encoder_outputs['encoder_out']
                encoder_padding_mask = encoder_outputs['encoder_padding_mask']
                decoder_input = input.new(bsz, decode_length+1, self.args.audio_num_mel_bins).fill_(self.padding_idx).float()
                output = input.new(bsz, 0, self.args.audio_num_mel_bins+1).fill_(self.padding_idx).float()
                hit_eos = input.new(bsz, 1).fill_(0).bool()
                stop_logits = input.new(bsz, 0).fill_(0).float()
                encdec_attn_logits = []
                if arch is not None: # in ws mode, arch is provided at run
                    num_heads = utils.get_num_heads(arch[2*self.args.enc_layers:])
                else: # in general mode, arch is defined at the begining
                    num_heads = self.num_heads
                for i in range(self.args.dec_layers):
                    encdec_attn_logits.append(input.new(bsz, num_heads[2*i], 0, max_input_len).fill_(0).float())
                    encdec_attn_logits.append(input.new(bsz, num_heads[2*i+1], 0, max_input_len).fill_(0).float())
                incremental_state = {}
                for step in range(decode_length):
                    decoder_output, attn_logits = model.forward_decoder(decoder_input[:, :step+1], encoder_out, encoder_padding_mask, incremental_state=incremental_state, arch=arch, layer_norm_training=layer_norm_training)
                    next_mel = decoder_output[:, -1:, :self.args.audio_num_mel_bins]
                    stop_logit = decoder_output[:, -1:, -1]
                    stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
                    output = torch.cat((output, decoder_output), dim=1)
                    for i in range(self.args.dec_layers):
                        encdec_attn_logits[2*i] = torch.cat((encdec_attn_logits[2*i], attn_logits[2*i]), dim=2)
                        encdec_attn_logits[2*i+1] = torch.cat((encdec_attn_logits[2*i+1], attn_logits[2*i+1]), dim=2)
                    decoder_input[:, step+1] = next_mel[:, -1]
                    this_hit_eos = hit_eos[:, -1:]
                    this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
                    hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)
            
            loss_output = self.loss(output, target)
            decoder_loss = loss_output['decoder loss']
            stop_loss = loss_output['stop loss']
            total_loss = decoder_loss + stop_loss
            encdec_attn = utils.select_attn(encdec_attn_logits)

            src_lengths = utils.move_to_cuda(sample['src_lengths']) - 1 # exclude eos
            src_lengths = utils.move_to_cuda(sample['src_lengths']) - 1 # exclude eos
            if target is not None:
                target_lengths = utils.move_to_cuda(sample['target_lengths'])
                target_padding_mask = target.abs().sum(-1).eq(self.padding_idx)
            else:
                hit_eos = hit_eos[:, 1:]
                target_lengths = (1.0 - hit_eos.float()).sum(dim=-1)
                target_padding_mask = output[:, :, :self.args.audio_num_mel_bins].abs().sum(-1).eq(self.padding_idx)
            src_padding_mask = input.eq(self.padding_idx) | input.eq(self.eos_idx)  # also exclude eos
            src_seg_mask = input.eq(self.seg_idx)
            focus_rate = utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
            phone_coverage_rate = utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask, target_padding_mask)
            attn_ks = src_lengths.float() / target_lengths.float()
            diagonal_focus_rate = utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths, src_padding_mask, target_padding_mask)

            ret = {
                'focus_rate': focus_rate.mean().data,
                'phone_coverage_rate': phone_coverage_rate.mean().data,
                'diagonal_focus_rate': diagonal_focus_rate.mean().data,
                'loss': total_loss.data,
                'nsamples': sample['nsamples'],
            }
            return ret
