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
parser.add_argument('--max_epochs', type=int, default=10000)
parser.add_argument('--max_updates', type=int, default=80000)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--arch', type=str, default='8 8 8 8 8 8 8 8')
parser.add_argument('--enc_layers', type=int, default=4)
parser.add_argument('--dec_layers', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--prenet_hidden_size', type=int, default=32)
parser.add_argument('--max_tokens', type=int, default=31250)
parser.add_argument('--max_sentences', type=int, default=None)
parser.add_argument('--stop_token_weight', type=float, default=5.0)
parser.add_argument('--lr', type=float, default=2.0)
parser.add_argument('--warmup_updates', type=int, default=4000)
parser.add_argument('--optimizer_adam_beta1', type=float, default=0.9)
parser.add_argument('--optimizer_adam_beta2', type=float, default=0.98)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--clip_grad_norm', type=float, default=1)
parser.add_argument('--attn_constraint', action='store_true')
parser.add_argument('--gta', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def main(args):
    logging.info('training on {} gpus'.format(torch.cuda.device_count()))
    logging.info('max tokens {} per gpu'.format(args.max_tokens))
    logging.info('max sentences {} per gpu'.format(args.max_sentences))
        
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
    
    saved_args, model_state_dict, epoch, global_step, optimizer_state_dict, best_valid_loss = utils.load(args.output_dir)
    if any([saved_args, model_state_dict, epoch, global_step, optimizer_state_dict]):
        logging.info('Found exist checkpoint with epoch %d and updates %d', epoch, global_step)

    if saved_args is not None:
        saved_args.__dict__.update(args.__dict__)
        args = saved_args
    task = tasks.LJSpeechTask(args)
    task.setup_task(model_state_dict, optimizer_state_dict)
    logging.info("param size = %d", utils.count_parameters(task.model))
    
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    else:
        max_epochs = float('inf')
    if args.max_updates is not None:
        max_updates = args.max_updates
    else:
        max_updates = float('inf')

    while epoch < max_epochs and global_step < max_updates:
        epoch += 1
        decoder_loss, stop_loss, loss, global_step = task.train(epoch=epoch, num_updates=global_step)
        logging.info('train %d global step %d decoder loss %.6f stop loss %.6f total loss %.6f',
                    epoch, global_step, decoder_loss, stop_loss, loss)
        decoder_loss, stop_loss, loss, fr, pcr, dfr = task.valid()
        logging.info('valid %d global step %d decoder loss %.6f stop loss %.6f total loss %.6f fr %.6f pcr %.6f dfr %.6f',
                    epoch, global_step, decoder_loss, stop_loss, loss, fr, pcr, dfr)
        is_best = False
        if loss < best_valid_loss:
            best_valid_loss = loss
            is_best = True
        if epoch % args.save_interval == 0:
            utils.save(args.output_dir, args, task.model, epoch, global_step, task.optimizer, best_valid_loss, is_best)
  

if __name__ == '__main__':
    main(args)
