from collections import defaultdict
import os
import sys
import shutil
import types
import numpy as np
import torch
import torch.nn.functional as F

class HParams(object):
    def __init__(self, argdict=None):
        if argdict:
            for k,v in argdict.items():
                setattr(self, k, v)


def move_to_cuda(tensor):
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def collate_tokens(values, pad_idx, left_pad=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_mels(values, pad_idx, left_pad=False, shift_right=False):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size, 80).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx
    

def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)

def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
      

def save(model_path, args, model, epoch, step, optimizer, best_valid_metric=None, is_best=True):
    if hasattr(model, 'module'):
        model = model.module
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'best_valid_metric': best_valid_metric,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    shutil.copyfile(filename, newest_filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)
  

def load(model_path):
    if os.path.isdir(model_path):
        newest_filename = os.path.join(model_path, 'checkpoint.pt')
    else:
        assert os.path.isfile(model_path), model_path
        newest_filename = model_path
    if not os.path.exists(newest_filename):
        return None, None, 0, 0, None, float('inf')
    state_dict = torch.load(newest_filename)
    args = state_dict['args']
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    best_valid_metirc = state_dict['best_valid_metric'] if state_dict['best_valid_metric'] is not None else float('inf')
    return args, model_state_dict, epoch, step, optimizer_state_dict, best_valid_metirc


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)



def unpack_dict_to_list(samples):
        samples_ = []
        bsz = samples.get('id').size(0)
        for i in range(bsz):
            res = {}
            for k, v in samples.items():
                try:
                    res[k] = v[i]
                except:
                    pass
            samples_.append(res)
        return samples_


def get_focus_rate(attn, src_padding_mask=None, tgt_padding_mask=None):
    ''' 
    attn: bs x L_t x L_s
    '''
    if src_padding_mask is not None:
        attn = attn * (1 - src_padding_mask.float())[:, None, :]
    
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    focus_rate = attn.max(-1).values.sum(-1)
    focus_rate = focus_rate / attn.sum(-1).sum(-1)
    return focus_rate



def get_phone_coverage_rate(attn, src_padding_mask=None, src_seg_mask=None, tgt_padding_mask=None):
    ''' 
    attn: bs x L_t x L_s
    '''
    src_mask = attn.new(attn.size(0), attn.size(-1)).bool().fill_(False)
    if src_padding_mask is not None:
        src_mask |= src_padding_mask
    if src_seg_mask is not None:
        src_mask |= src_seg_mask
    
    attn = attn * (1 - src_mask.float())[:, None, :]
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]
    
    phone_coverage_rate = attn.max(1).values.sum(-1)
    phone_coverage_rate = phone_coverage_rate / (1 - src_mask.float()).sum(-1)
    return phone_coverage_rate


def get_diagonal_focus_rate(attn, attn_ks, target_len, src_padding_mask=None, tgt_padding_mask=None, band_mask_factor=5):
    ''' 
    attn: bx x L_t x L_s
    attn_ks: shape: tensor with shape [batch_size], input_lens/output_lens
    
    diagonal: y=k*x (k=attn_ks, x:output, y:input)
    1 0 0
    0 1 0
    0 0 1
    y>=k*(x-width) and y<=k*(x+width):1
    else:0
    '''
    #width = min(target_len/band_mask_factor, 50)
    width1 = target_len / band_mask_factor
    width2 = target_len.new(target_len.size()).fill_(50)
    width = torch.where(width1 < width2, width1, width2).float()
    base = torch.ones(attn.size()).to(attn.device)
    zero = torch.zeros(attn.size()).to(attn.device)
    x = torch.arange(0, attn.size(1)).to(attn.device)[None, :, None].float() * base
    y = torch.arange(0, attn.size(2)).to(attn.device)[None, None, :].float() * base
    cond = (y - attn_ks[:, None, None] * x)
    cond1 = cond + attn_ks[:, None, None] * width[:, None, None]
    cond2 = cond - attn_ks[:, None, None] * width[:, None, None]
    mask1 = torch.where(cond1 < 0, zero, base)
    mask2 = torch.where(cond2 > 0, zero, base)
    mask = mask1 * mask2

    if src_padding_mask is not None:
        attn = attn * (1 - src_padding_mask.float())[:, None, :]
    if tgt_padding_mask is not None:
        attn = attn * (1 - tgt_padding_mask.float())[:, :, None]

    diagonal_attn = attn * mask
    diagonal_focus_rate = diagonal_attn.sum(-1).sum(-1) / attn.sum(-1).sum(-1)
    return diagonal_focus_rate


def generate_arch(n, layers, num_ops=10):
    def _get_arch():
        arch = [np.random.randint(1, num_ops+1) for _ in range(layers)]
        return arch
    archs = [_get_arch() for i in range(n)]
    return archs


def parse_arch_to_seq(arch):
    seq = [op for op in arch]
    return seq


def parse_seq_to_arch(seq):
    arch = [idx for idx in seq]
    return arch


def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


def select_attn(attn_logits):
    num_layers = len(attn_logits)
    encdec_attn = []
    for i in range(num_layers):
        attn = attn_logits[i].softmax(-1)
        indices = attn.max(-1).values.sum(-1).argmax(-1)
        attn = attn.gather(1, indices[:, None, None, None].repeat(1, 1, attn.size(-2), attn.size(-1))).squeeze(1) # select max head per layer
        encdec_attn.append(attn)
    encdec_attn = torch.stack(encdec_attn, dim=1)
    indices = encdec_attn.max(-1).values.sum(-1).argmax(-1)
    encdec_attn = encdec_attn.gather(1, indices[:, None, None, None].repeat(1, 1, encdec_attn.size(-2), encdec_attn.size(-1))).squeeze(1) # select max layer
    return encdec_attn


def get_num_heads(arch):
    num_heads = []
    for i in range(len(arch)):
        op = arch[i]
        if op <= 7 or op == 11: 
            num_heads.append(1)
        elif op == 8:
            num_heads.append(2)
        elif op == 9:
            num_heads.append(4)
        elif op == 10:
            num_heads.append(8)
    return num_heads
