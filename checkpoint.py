import os
import torch
from tensorflow.io import gfile
import numpy as np


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
            tensor_value = tensor_value.reshape(feat_dim, num_heads*head_dim)
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['q', 'k', 'v']:
            # for multi head attention q/k/v bias
            tensor_value = tensor_value.reshape(-1)
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            # for multi head attention out weight
            num_heads, head_dim, feat_dim = tensor_value.shape
            tensor_value = tensor_value.reshape(num_heads*head_dim, feat_dim)
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.permute(3, 2, 0, 1)

        # print("{}: {}".format(torch_key, tensor_value.shape))
        state_dict[torch_key] = tensor_value
    state_dict['pos_embedding'] = state_dict['encoder.pos_embedding.pos_embedding']
    state_dict.pop('encoder.pos_embedding.pos_embedding')
    return state_dict

