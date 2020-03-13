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
import nao_controller
import operations

parser = argparse.ArgumentParser(description='SemiNAS Search for TTS')

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='data/LJSpeech-1.1')
parser.add_argument('--raw_data', action='store_true', default=False)
parser.add_argument('--max_epochs', type=int, default=1)
parser.add_argument('--max_updates', type=int, default=None)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--enc_layers', type=int, default=6)
parser.add_argument('--dec_layers', type=int, default=6)
parser.add_argument('--hidden_size', type=int, default=64)
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
parser.add_argument('--save_interval', type=int, default=None)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--arch_pool', type=str, default=None)
parser.add_argument('--reward', type=str, default='0 0 1 0')

parser.add_argument('--controller_iterations', type=int, default=3)
parser.add_argument('--controller_seed_arch', type=int, default=100)
parser.add_argument('--controller_random_arch', type=int, default=4000)
parser.add_argument('--controller_up_sample_ratio', type=int, default=None)
parser.add_argument('--controller_new_arch', type=int, default=100)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_mlp_layers', type=int, default=2)
parser.add_argument('--controller_hidden_size', type=int, default=16)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=64)
parser.add_argument('--controller_dropout', type=float, default=0.1)
parser.add_argument('--controller_weight_decay', type=float, default=1e-4)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_pretrain_epochs', type=int, default=10000)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_clip_grad_norm', type=float, default=5.0)
args = parser.parse_args()


utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def main(args):
    logging.info('training on {} gpus'.format(torch.cuda.device_count()))
    logging.info('max tokens {} per gpu'.format(args.max_tokens))
    logging.info('max sentences {} per gpu'.format(args.max_sentences if args.max_sentences is not None else 'None'))
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    args.child_num_ops = len(operations.OPERATIONS_ENCODER)
    args.controller_vocab_size = 1 + args.child_num_ops
    args.controller_source_length = args.enc_layers + args.dec_layers
    args.controller_encoder_length = args.controller_decoder_length = args.controller_source_length
    tasks.set_ljspeech_hparams(args)
    logging.info("args = {}".format(args))
    
    if args.arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.arch_pool) as f:
            archs = f.read().splitlines()
            child_arch_pool = [list(map(int, arch.strip().split())) for arch in archs]
    else:
        child_arch_pool = None

    task = tasks.LJSpeechTask(args)
    task.setup_task(ws=True)
    logging.info("Model param size = %d", utils.count_parameters(task.model))

    controller = nao_controller.NAO(args)
    logging.info("Encoder-Predictor-Decoder param size = %d", utils.count_parameters(controller))

    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.enc_layers + args.dec_layers, args.child_num_ops)

    arch_pool = []
    num_updates = 0
    r_c = list(map(float, args.reward.strip().split()))
    for controller_iteration in range(args.controller_iterations+1):
        logging.info('Iteration %d', controller_iteration+1)
        arch_pool += child_arch_pool
        for epoch in range(1, args.max_epochs * len(arch_pool) + 1):
            decoder_loss, stop_loss, loss, num_updates = task.train(epoch=epoch, num_updates=num_updates, arch_pool=arch_pool)
            lr = task.scheduler.get_lr()
            logging.info('epoch %d updates %d lr %.6f decoder loss %.6f stop loss %.6f loss %.6f', epoch, num_updates, lr, decoder_loss, stop_loss, loss)
        frs, pcrs, dfrs, losses = task.valid_for_search(None, gta=False, arch_pool=arch_pool, layer_norm_training=True)
        reward = [r_c[0] * fr + r_c[1] * pcr + r_c[2] * dfr - r_c[3] * loss for fr, pcr, dfr, loss in zip(frs, pcrs, dfrs, losses)]
        arch_pool_valid_perf = reward
        
        arch_pool_valid_perf_sorted_indices = np.argsort(arch_pool_valid_perf)[::-1]
        arch_pool = list(map(lambda x:arch_pool[x], arch_pool_valid_perf_sorted_indices))
        arch_pool_valid_perf = list(map(lambda x:arch_pool_valid_perf[x], arch_pool_valid_perf_sorted_indices))
        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(controller_iteration)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(controller_iteration)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_perf):
                    arch = ' '.join(map(str, arch))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if controller_iteration == args.controller_iterations:
            break
                            
        # Train Encoder-Predictor-Decoder
        logging.info('Train Encoder-Predictor-Decoder')
        inputs = list(map(lambda x: utils.parse_arch_to_seq(x), arch_pool))
        min_val = min(arch_pool_valid_perf)
        max_val = max(arch_pool_valid_perf)
        targets = list(map(lambda x: (x - min_val) / (max_val - min_val), arch_pool_valid_perf))

        # Pre-train NAO
        logging.info('Pre-train EPD')
        controller.build_dataset('train', inputs, targets, True)
        controller.build_queue('train')
        for epoch in range(1, args.controller_pretrain_epochs+1):
            loss, mse, ce = controller.train_epoch('train')
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)
        logging.info('Finish pre-training EPD')

        # Generate synthetic data
        logging.info('Generate synthetic data for EPD')
        synthetic_inputs, synthetic_targets = controller.generate_synthetic_data(inputs, args.controller_random_arch)
        if args.controller_up_sample_ratio:
            all_inputs = inputs * args.controller_up_sample_ratio + synthetic_inputs
            all_targets = targets * args.controller_up_sample_ratio + synthetic_targets
        else:
            all_inputs = inputs + synthetic_inputs
            all_targets = targets + synthetic_targets
        # Train NAO
        logging.info('Train EPD')
        controller.build_dataset('train', all_inputs, all_targets, True)
        controller.build_queue('train')
        for epoch in range(1, args.controller_epochs+1):
            loss, mse, ce = controller.train_epoch('train')
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)
        logging.info('Finish training EPD')


        # Generate new archs
        new_archs = []
        max_step_size = 100
        predict_step_size = 0
        # get top 100 from true data and synthetic data
        topk_indices = np.argsort(all_targets)[:100]
        topk_archs = list(map(lambda x:all_inputs[x], topk_indices))
        controller.build_dataset('infer', topk_archs, None, False)
        controller.build_queue('infer', batch_size=len(topk_archs), shuffle=False)
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 0.1
            logging.info('Generate new architectures with step size %.2f', predict_step_size)
            new_arch = controller.infer('infer', predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in inputs and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break

        child_arch_pool = list(map(lambda x: utils.parse_seq_to_arch(x), new_archs))
        logging.info("Generate %d new archs", len(child_arch_pool))

    logging.info('Finish Searching')


if __name__ == '__main__':
    main(args)
