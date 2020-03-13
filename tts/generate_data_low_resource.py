import os
import sys
import json
import logging
import shutil
import argparse
import random
import numpy as np
import pandas as pd
import utils
import logging
import subprocess
from text_encoder import TokenTextEncoder
from preprocessor import _process_utterance
import audio

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

parser = argparse.ArgumentParser(description='TTS')
# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='data/LJSpeech-1.1/raw')
parser.add_argument('--output_dir', type=str, default='data/LJSpeech-1.1/processed_low_resource_1k5')
parser.add_argument('--train_num', type=int, default=1500)
parser.add_argument('--train_time', type=int, default=None)
parser.add_argument('--test_num', type=int, default=100)
parser.add_argument('--valid_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


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


def split_train_test_set(data_dir, train_num, valid_num, test_num, train_time=None):
    data_file_name = 'metadata_phone.csv'
    data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
    total_num = len(data_df.index)
    indices = [i for i in np.arange(0, total_num)]
    np.random.shuffle(indices)
    if valid_num is not None and valid_num > 0:
        valid_uttids = indices[:valid_num]
    else:
        valid_num = 0
        valid_uttids = []
    if test_num is not None and test_num > 0:
        test_uttids = indices[valid_num:valid_num+test_num]
    else:
        test_num = 0
        test_uttids = []
    if train_num is not None and train_num > 0:
        train_uttids = indices[valid_num+test_num:valid_num+test_num+train_num]
    elif train_time is not None and train_time > 0:
        offset = valid_num+test_num
        train_uttids = []
        total_time = 0
        idx = 0
        while total_time < train_time:
            uttid = indices[offset+idx]
            r = data_df.iloc[uttid]
            wav_file = os.path.join(data_dir, 'wavs', r['wav'] + '.wav')
            res = subprocess.getoutput('sox {} -n stat'.format(wav_file))
            dur = float(res.splitlines()[1].split()[-1])
            total_time += dur
            train_uttids.append(uttid)
            idx += 1
    else:
        train_uttids = indices[valid_num+test_num:]

    logging.info(">>train {}".format(len(train_uttids)))
    logging.info(">>valid {}".format(len(valid_uttids)))
    logging.info(">>test {}".format(len(test_uttids)))
    logging.info(">>total {}".format(len(list(set(test_uttids + train_uttids + valid_uttids)))))
    return sorted(train_uttids), sorted(valid_uttids), sorted(test_uttids)


def build_phone_encoder(data_dir):
    phone_list_file = os.path.join(data_dir, 'phone_set.json')
    phone_list = json.load(open(phone_list_file))
    return TokenTextEncoder(None, vocab_list=phone_list)


def process_wav(data_dir, filename, args):
    media_file = os.path.join(data_dir, 'wavs', filename + '.wav')
    wav_data, mel_data = _process_utterance(media_file, args)
    return wav_data, mel_data


def produce_result(utt_id, text, phone, mel, encoder):
    phones = phone.split(" ")
    phones += ['|']
    phones = ' '.join(phones)
    try:
        utt_id = int(utt_id)
        phone_encoded = encoder.encode(phones) + [encoder.eos()]
        mel = mel.T #(T*80)
    except Exception as e:
        logging.info('{} {}'.format(e, text))
        return None

    return utt_id, text, phone_encoded, mel


def read_data(data_dir, encoder, utt_ids=None):
    data_df = pd.read_csv(os.path.join(data_dir, 'metadata_phone.csv'))
        
    texts, phones, mels = [], [], []
    if utt_ids is None:
        for idx, r in data_df.iterrows():
            utt_id, text, phone, mel = produce_result(idx, r['txt2'], r['phone2'], process_wav(data_dir, r['wav'], args)[1], encoder)
            texts.append(text)
            phones.append(phone)
            mels.append(mel)
    else:
        for utt_id in utt_ids:
            r = data_df.iloc[utt_id]
            utt_id, text, phone, mel = produce_result(utt_id, r['txt2'], r['phone2'], process_wav(data_dir, r['wav'], args)[1], encoder)
            texts.append(text)
            phones.append(phone)
            mels.append(mel)
    return texts, phones, mels


def write_data(data_dir, prefix, utt_ids, texts, phones, mels):
    logging.info('Generating {} data'.format(prefix))
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, prefix+'.idx'), 'w') as f1:
        with open(os.path.join(data_dir, prefix+'.text'), 'w') as f2:
            with open(os.path.join(data_dir, prefix+'.phone'), 'w') as f3:
                with open(os.path.join(data_dir, prefix+'.mel'), 'wb') as f4:
                    for i, (utt_id, text, phone) in enumerate(zip(utt_ids, texts, phones)):
                        f1.write('{}\n'.format(utt_id))
                        f2.write('{}\n'.format(text))
                        f3.write('{}\n'.format(' '.join(map(str, phone))))
                        if (i+1) % 100 == 0:
                            logging.info('Generated {}/{}'.format(i+1, len(utt_ids)))
                    np.save(f4, mels)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    set_ljspeech_hparams(args)
    train_uttids, valid_uttids, test_uttids = split_train_test_set(args.data_dir, args.train_num, args.valid_num, args.test_num, args.train_time)
    encoder = build_phone_encoder(args.data_dir)
    train_texts, train_phones, train_mels = read_data(args.data_dir, encoder, train_uttids)
    valid_texts, valid_phones, valid_mels = read_data(args.data_dir, encoder, valid_uttids)
    test_texts, test_phones, test_mels = read_data(args.data_dir, encoder, test_uttids)
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    write_data(output_dir, 'train', train_uttids, train_texts, train_phones, train_mels)
    write_data(output_dir, 'valid', valid_uttids, valid_texts, valid_phones, valid_mels)
    write_data(output_dir, 'test', test_uttids, test_texts, test_phones, test_mels)
    shutil.copy(os.path.join(args.data_dir, 'phone_set.json'), os.path.join(args.output_dir, 'phone_set.json'))


if __name__ == "__main__":
    main(args)
