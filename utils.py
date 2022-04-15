import os
import torch
from tensorflow.io import gfile
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def load_checkpoint(path):
    """ Load weights from a given checkpoint path in npz/pth """
    if path.endswith('npz'):
        keys, values = load_jax(path)
        state_dict = convert_jax_pytorch(keys, values)
    elif path.endswith('pth'):
        state_dict = torch.load(path)['state_dict']
    else:
        raise ValueError("checkpoint format {} not supported yet!".format(path.split('.')[-1]))

    return state_dict


def load_jax(path):
    """ Loads params from a npz checkpoint previously stored with `save()` in jax implemetation """
    with gfile.GFile(path, 'rb') as f:
        ckpt_dict = np.load(f, allow_pickle=False)
        keys, values = zip(*list(ckpt_dict.items()))
    return keys, values


def save_jax_to_pytorch(jax_path, save_path):
    model_name = jax_path.split('/')[-1].split('.')[0]
    keys, values = load_jax(jax_path)
    state_dict = convert_jax_pytorch(keys, values)
    checkpoint = {'state_dict': state_dict}
    torch.save(checkpoint, os.path.join(save_path, model_name + '.pth'))


def replace_names(names):
    """ Replace jax model names with pytorch model names """
    new_names = []
    for name in names:
        if name == 'Transformer':
            new_names.append('encoder')
        elif name == 'encoder_norm':
            new_names.append('ln')
        elif 'encoderblock' in name:
            num = name.split('_')[-1]
            new_names.append('layers')
            new_names.append(num)
        elif 'LayerNorm' in name:
            num = name.split('_')[-1]
            if num == '0':
                new_names.append('ln_{}'.format(1))
            elif num == '2':
                new_names.append('ln_{}'.format(2))
        elif 'MlpBlock' in name:
            new_names.append('mlp')
        elif 'Dense' in name:
            num = name.split('_')[-1]
            new_names.append('linear_{}'.format(int(num) + 1))
        elif 'MultiHeadDotProductAttention' in name:
            new_names.append('self_attention')
        elif name == 'kernel' or name == 'scale':
            new_names.append('weight')
        elif name == 'bias':
            new_names.append(name)
        elif name == 'posembed_input':
            new_names.append('pos_embedding')
        elif name == 'pos_embedding':
            new_names.append('pos_embedding')
        elif name == 'embedding':
            new_names.append('embedding')
        elif name == 'head':
            new_names.append('head')
        elif name == 'cls':
            new_names.append('cls_token')
        elif name == 'key':
            new_names.append('k')
        elif name == 'query':
            new_names.append('q')
        elif name == 'value':
            new_names.append('v')
        else:
            new_names.append(name)
    return new_names


def convert_jax_pytorch(keys, values):
    """ Convert jax model parameters with pytorch model parameters """
    state_dict = {}
    for key, value in zip(keys, values):

        # convert name to torch names
        names = key.split('/')
        torch_names = replace_names(names)
        torch_key = '.'.join(w for w in torch_names)

        # convert values to tensor and check shapes
        tensor_value = torch.tensor(value, dtype=torch.float)
        # check shape
        num_dim = len(tensor_value.shape)

        if num_dim == 1:
            tensor_value = tensor_value.squeeze()
        elif num_dim == 2 and torch_names[-1] == 'weight':
            # for normal weight, transpose it
            tensor_value = tensor_value.T
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] in ['q', 'k', 'v']:
            feat_dim, num_heads, head_dim = tensor_value.shape
            # for multi head attention q/k/v weight
            tensor_value = tensor_value.flatten(1).T
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['q', 'k', 'v']:
            # for multi head attention q/k/v bias
            tensor_value = tensor_value.reshape(-1)
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            # for multi head attention out weight
            num_heads, head_dim, feat_dim = tensor_value.shape
            tensor_value = tensor_value.reshape(num_heads*head_dim, feat_dim).T
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.permute(3, 2, 0, 1)

        # print("{}: {}".format(torch_key, tensor_value.shape))
        state_dict[torch_key] = tensor_value
    state_dict['pos_embedding'] = state_dict['encoder.pos_embedding.pos_embedding']
    state_dict.pop('encoder.pos_embedding.pos_embedding')
    return state_dict

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):

    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def get_parameter_num(model):
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    return num_trainable_parameters