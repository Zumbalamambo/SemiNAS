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
from text_encoder import TokenTextEncoder
from preprocessor import _process_utterance
import audio

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

parser = argparse.ArgumentParser(description='TTS')
# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='data/LJSpeech-1.1/raw')
parser.add_argument('--output_dir', type=str, default='data/LJSpeech-1.1/processed_robust100')
parser.add_argument('--valid_phone_file', type=str, default=None)
parser.add_argument('--valid_text_file', type=str, default=None)
parser.add_argument('--test_phone_file', type=str, default=None)
parser.add_argument('--test_text_file', type=str, default=None)
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


def read_lj_data(data_dir, encoder):
    data_df = pd.read_csv(os.path.join(data_dir, 'metadata_phone.csv'))
        
    texts, phones, mels = [], [], []
    for idx, r in data_df.iterrows():
        utt_id, text, phone, mel = produce_result(idx, r['txt2'], r['phone2'], process_wav(data_dir, r['wav'], args)[1], encoder)
        texts.append(text)
        phones.append(phone)
        mels.append(mel)
    return texts, phones, mels


def read_custom_data(data_dir, phone_file, text_file, encoder):
    with open(os.path.join(data_dir, phone_file), 'r') as f:
        phones = f.read().splitlines()
    with open(os.path.join(data_dir, text_file), 'r') as f:
        texts = f.read().splitlines()
    assert len(phones) == len(texts)
    # process phones
    encoded_phones = []
    for phone in phones:
        phone = phone.split(" ")
        phone += ['|']
        phone = ' '.join(phone)
        phone_encoded = encoder.encode(phone) + [encoder.eos()]
        encoded_phones.append(phone_encoded)

    return texts, encoded_phones


def write_data(data_dir, prefix, utt_ids=None, texts=None, phones=None, mels=None):
    logging.info('Generating {} data'.format(prefix))
    os.makedirs(data_dir, exist_ok=True)
    if utt_ids:
        with open(os.path.join(data_dir, prefix+'.idx'), 'w') as f:
            for i, utt_id in enumerate(utt_ids):
                f.write('{}\n'.format(utt_id))
    if texts:
        with open(os.path.join(data_dir, prefix+'.text'), 'w') as f:
            for i, text in enumerate(texts):
                f.write('{}\n'.format(text))
    if phones:
        with open(os.path.join(data_dir, prefix+'.phone'), 'w') as f:
            for i, phone in enumerate(phones):
                f.write('{}\n'.format(' '.join(map(str, phone))))
    if mels:
        with open(os.path.join(data_dir, prefix+'.mel'), 'wb') as f:
            np.save(f, mels)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    set_ljspeech_hparams(args)
    encoder = build_phone_encoder(args.data_dir)
    train_texts, train_phones, train_mels = read_lj_data(args.data_dir, encoder)
    valid_texts, valid_phones = read_custom_data(args.data_dir, args.valid_phone_file, args.valid_text_file, encoder)
    test_texts, test_phones = read_custom_data(args.data_dir, args.test_phone_file, args.test_text_file, encoder)
    train_uttids = list(range(len(train_texts)))
    valid_uttids = list(range(len(valid_texts)))
    test_uttids = list(range(len(test_texts)))
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    write_data(output_dir, 'train', train_uttids, train_texts, train_phones, train_mels)
    write_data(output_dir, 'valid', valid_uttids, valid_texts, valid_phones)
    write_data(output_dir, 'test', test_uttids, test_texts, test_phones)
    shutil.copy(os.path.join(args.data_dir, 'phone_set.json'), os.path.join(args.output_dir, 'phone_set.json'))


if __name__ == "__main__":
    main(args)
