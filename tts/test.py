import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import utils
import logging
import argparse
import torch.backends.cudnn as cudnn
import tasks

parser = argparse.ArgumentParser(description='TTS')

# Basic model parameters.
parser.add_argument('--debug', action='store_true')
parser.add_argument('--data_dir', type=str, default='data/LJSpeech-1.1')
parser.add_argument('--raw_data', action='store_true')
parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
parser.add_argument('--n', type=int, default=None)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--max_tokens', type=int, default=31250)
parser.add_argument('--max_sentences', type=int, default=None)
parser.add_argument('--gta', action='store_true')
parser.add_argument('--attn_constraint', action='store_true')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def main(args):
    if not torch.cuda.is_available():
        logging.info('No gpu device available')
        sys.exit(1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    tasks.set_ljspeech_hparams(args)
    logging.info("args = %s", args)
    
    saved_args, model_state_dict, epoch, global_step, optimizer_state_dict, best_valid_loss = utils.load(args.checkpoint_path)
    if any([saved_args, model_state_dict, epoch, global_step, optimizer_state_dict]):
        logging.info('Found exist checkpoint with epoch %d and updates %d', epoch, global_step)

    if saved_args is not None:
        saved_args.__dict__.update(args.__dict__)
        args = saved_args
    task = tasks.LJSpeechTask(args)
    task.setup_task(model_state_dict, optimizer_state_dict)
    task.infer(split=args.split, n=args.n)
  

if __name__ == '__main__':
    main(args)