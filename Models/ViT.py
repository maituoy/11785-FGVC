import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

from utils import load_checkpoint, resize_pos_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        return x

class MLPBlock(nn.Sequential):

    def __init__(self, in_dim, mlp_dim, dropout=0):
        super().__init__()
        
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

class EncoderBlock(nn.Module):

    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout):
        super().__init__()
        
        self.num_heads = num_heads

        # Components of the first half block
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = Attention(hidden_dim, 
                                        num_heads, 
                                        attn_drop=attention_dropout
                                        )
        self.dropout = nn.Dropout(dropout)

        # Components of the second half block
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x):

        input_x = x.clone()
        x = self.ln_1(x)
        x = self.self_attention(x)
        x = self.dropout(x)
        x += input_x

        y = self.ln_2(x)
        y = self.mlp(y)
        y += x

        return y

class Encoder(nn.Module):

    def __init__(self, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        layers = []

        for i in range(num_layers):
            layers.append(EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout))

        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)

        return x
        

class ViT(nn.Module):

    def __init__(
        self, 
        img_size, 
        patch_size, 
        num_layers, 
        num_heads, 
        hidden_dim, 
        mlp_dim, 
        dropout=0., 
        attention_dropout=0., 
        num_classes=1000
        ):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Conv2d(3, hidden_dim, patch_size, patch_size)

        seq_length = (img_size // patch_size) ** 2

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        seq_length += 1

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        

        self.encoder = Encoder(num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)

        self.head = nn.Linear(hidden_dim, num_classes)

        self.seq_length = seq_length

    def forward(self, x):

        n, c, h, w = x.shape
        p = self.patch_size

        n_h = h // p
        n_w = w // p

        # Patching and flatting the image based on patch size
        x = self.embedding(x)
        x = x.reshape(n, self.hidden_dim, n_h*n_w)
        x = x.permute(0, 2, 1)

        # Add a class token to each data
        batch_cls_token = self.cls_token.expand(n, -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        # Add position embedding
        x += self.pos_embedding

        # Feed data into the encoder
        x = self.encoder(x)

        # Only the first value is used for classification
        x = x[:, 0]

        # Send x to the classificaiton layer
        x = self.head(x)

        return x


def vit_s16(config, pretrained=False, **kwargs):

    input_size = config.data.input_size
    model = ViT(
        img_size = input_size,
        patch_size = 16,
        num_layers = 12,
        num_heads = 6,
        hidden_dim = 384,
        mlp_dim = 1536
    )

    if pretrained:
        state_dict = load_checkpoint(config.model.pretrained.path)

        if input_size == 224:
            model.load_state_dict(state_dict)
        
        elif input_size > 224:
            posemb = state_dict['pos_embedding']
            posemb_new = model.pos_embedding
            pos_emb = resize_pos_embed(posemb, posemb_new)
            state_dict['pos_embedding'] = pos_emb
            model.load_state_dict(state_dict)
    
    return model


