import argparse
import json
import utils

parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str, default=None)
parser.add_argument('--output', type=str, default='net.config')
parser.add_argument('--width_stages', type=str, default='32,56,112,128,256,432')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')

args = parser.parse_args()

args.width_stages = [int(val) for val in args.width_stages.split(',')]
args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
args.stride_stages = [int(val) for val in args.stride_stages.split(',')]

model_config = utils.build_model_config(args.arch, args.width_stages, args.n_cell_stages, args.stride_stages, 0)
model_config['name'] = 'SemiNASNets'
with open(args.output, 'w') as f:
    json.dump(model_config, f)
