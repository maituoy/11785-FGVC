import os
import yaml
from addict import Dict as adict
from typing import List, Tuple, Union
from datetime import datetime

config = adict()

#Train aug
## EMA related parameters
config.train.model_ema = False
config.train.model_ema_decay = 0.9999
config.train.model_ema_force_cpu = False
config.train.model_ema_eval = False


## label_smoothing
config.train.smoothing = 0.1

## mixup params
config.train.mixup = 0.8
config.train.cutmix = 1.0
config.train.cutmix_minmax = None
config.train.mixup_prob = 1.0
config.train.mixup_switch_prob = 0.5
config.train.mixup_mode = "batch"


# General configs
config.exp = None
config.seed = 11785
config.world_size = 1
config.use_amp = True
config.output_dir = './experiments'
config.sync_bn = False
config.log_interval = 50
config.eval_metric = 'val_top1'

# Dataset
config.data.root = './data'
config.data.name = 'CUB2011'
config.data.image_size = 256
config.data.input_size = 224
config.data.batch_size = 32
config.data.num_workers = 8
config.data.sampler.name = None
config.data.sampler.param = dict()

# Optimizer
config.optim.name = 'adamW'
config.optim.lr = 0.002
config.optim.weight_decay = 1e-4
config.optim.param = dict()
config.optim.sched = 'cosine'
config.optim.epochs = 100
config.optim.warmup_epochs = 5

# Criterion
config.loss.name = 'ce'
config.loss.label_smoothing = 0.0

# Model
config.model.pretrained.path = None
config.model.pretrained.pretrain = True
config.model.backbone.name = 'resnet50'
config.model.backbone.param = dict()
config.model.head.num_classes = 200


def update_cfg(cfg: adict, cfg_yaml: str, cfg_argv: List[str]) -> None:
    # Update default cfg with given yaml file and argv list
    with open(cfg_yaml) as f:
        user_cfg = yaml.load(f, Loader=yaml.FullLoader)
        _recursive_update(cfg, adict(user_cfg))
    update_cfg_from_argv(cfg, cfg_argv)


def _recursive_update(_cfg, _user_cfg):
    """ 
    Recursively update the addict
    :param _cfg: The addict to be updated
    :param _user_cfg: The source dict given by user
    """
    for key, value in _user_cfg.items():
        if key not in _cfg:
            raise AttributeError(f'key is restricted among {list(_cfg.keys())},'
                                 f' got {key} instead!')
        if isinstance(_cfg[key], dict) and not isinstance(_cfg[key], adict):  # Unrestricted cfg like param
            assert isinstance(value, dict), f'value for {key} must be a dict, got {value} instead'
            _cfg[key] = adict(value)
        elif not isinstance(value, dict):
            if isinstance(value, (list, tuple)):  # Unpack list/tuple of dict into adict
                value = adict(tmp=value)['tmp']
            elif isinstance(value, str) and value.strip().lower() in ['none', 'null']:
                value = None
            _cfg[key] = value
        else:
            _recursive_update(_cfg[key], value)


# Update cfg from agrv for override
def update_cfg_from_argv(cfg: adict,
                         cfg_argv: List[str],
                         delimiter: str = '=') -> None:
    r""" Update global cfg with list from argparser
    Args:
        cfg: the cfg to be updated by the argv
        cfg_argv: the new config list, like ['epoch=10', 'save.last=False']
        dilimeter: the dilimeter between key and value of the given config
    """

    def resolve_cfg_with_legality_check(keys: List[str]) -> Tuple[adict, str]:
        r""" Resolve the parent and leaf from given keys and check their legality.
        Args:
            keys: The hierarchical keys of global cfg
        Returns:
            the resolved parent adict obj and its legal key to be upated.
        """

        obj, obj_repr = cfg, 'cfg'
        for idx, sub_key in enumerate(keys):
            if not isinstance(obj, adict) or sub_key not in obj:
                raise ValueError(f'Undefined attribute "{sub_key}" detected for "{obj_repr}"')
            if idx < len(keys) - 1:
                obj = obj.get(sub_key)
                obj_repr += f'.{sub_key}'
        return obj, sub_key

    for str_argv in cfg_argv:
        item = str_argv.split(delimiter, 1)
        assert len(item) == 2, "Error argv (must be key=value): " + str_argv
        key, value = item
        obj, leaf = resolve_cfg_with_legality_check(key.split('.'))
        obj[leaf] = eval(add_quotation_to_string(value))


def is_number_or_bool_or_none(x: str):
    r""" Return True if the given str represents a number (int or float) or bool
    """

    try:
        float(x)
        return True
    except ValueError:
        return x in ['True', 'False', 'None']


def add_quotation_to_string(s: str,
                            split_chars: List[str] = None) -> str:
    r""" For eval() to work properly, all string must be added quatation.
         Example: '[[train,3],[val,1]' -> '[["train",3],["val",1]'
    Args:
        s: the original value string
        split_chars: the chars that mark the split of the string
    Returns:
        the quoted value string
    """

    if split_chars is None:
        split_chars = ['[', ']', '{', '}', ',', ' ']
        if '{' in s and '}' in s:
            split_chars.append(':')
    s_mark, marker = s, chr(1)
    for split_char in split_chars:
        s_mark = s_mark.replace(split_char, marker)

    s_quoted = ''
    for value in s_mark.split(marker):
        if len(value) == 0:
            continue
        st = s.find(value)
        if is_number_or_bool_or_none(value):
            s_quoted += s[:st] + value
        elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
            s_quoted += s[:st] + value
        else:
            s_quoted += s[:st] + '"' + value + '"'
        s = s[st + len(value):]

    return s_quoted + s

def preprocess_cfg(cfg, local_rank):
    # local rank
    cfg.local_rank = local_rank

    # out_dir
    if cfg.local_rank == 0:
        cfg.output_dir = os.path.join(cfg.output_dir, cfg.data.name, cfg.exp + f'_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        os.makedirs(cfg.output_dir, exist_ok=True)
    
    # data dir 
    cfg.data.root = os.path.join(cfg.data.root, cfg.data.name)

