import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
#from ldm.modules.diffusionmodules.util import zero_module

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def prepare_mask(mask, h, w):
    N = mask.shape[0]
    rescaled_mask = torch.zeros(N, mask.shape[1]//(64//h), mask.shape[2]//(64//w))
    for i, m in enumerate(mask):
        h_idx, w_idx = torch.where(m==1)
        # patchify
        h_idx, w_idx = h_idx//(64//h), w_idx//(64//w)
        rescaled_mask[i][h_idx, w_idx] = 1
    
    rescaled_mask = rescaled_mask.view(N, -1)
    attn_mask = torch.zeros(N, rescaled_mask.shape[1], rescaled_mask.shape[1])
    attn_mask.requires_grad = False
    for i, m in enumerate(attn_mask):
        # get unzero index
        indices = torch.where(rescaled_mask[i]==1)[0]
        attn_mask[i,indices,:] = 1
        attn_mask[i,:,indices] = 1
    attn_mask = (attn_mask == 1)
    return attn_mask


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask):
        x = self.norm(x)
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            max_neg_value = -torch.finfo(dots.dtype).max
            mask = mask.unsqueeze(1).repeat(1,h,1,1)
            # dots.masked_fill_(~mask, max_neg_value)
            
            # fill with much small numbers but not max_neg
            min_val = torch.min(dots).detach()
            dots.masked_fill_(~mask, min_val)
            
        
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, mask):
        for i, (attn, ff) in enumerate(self.layers):
            if mask is not None:
                x = attn(x, mask[i]) + x
            else:
                x = attn(x, mask) + x
            x = ff(x) + x
        return x

class PMSAViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim=512, channels = 320, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.depth = depth

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 128 is the reduced channels
        patch_dim = 128 * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels,128,kernel_size=3,stride=1,padding=1),
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.h = image_height // patch_height
        self.w = image_width // patch_width
        
        self.pos_embedding = posemb_sincos_2d(
            h = self.h,
            w = self.w,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_out = nn.Sequential(
            Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_height, p2 = patch_width, h = image_size//patch_size),
            nn.Conv2d(128,channels,kernel_size=3,stride=1,padding=1),
        )

    def forward(self, img, mask=None):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # convert pose mask (4,64,64) to attn mask (4,32*32,32*32)
        attn_mask = None
        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                masks = torch.split(mask, x.shape[0])
                attn_mask = [prepare_mask(m, self.h, self.w).to(device) for m in masks]
                assert self.depth == len(attn_mask)
            else:
                attn_mask = [prepare_mask(mask, self.h, self.w).to(device)]
            
        x = self.transformer(x, attn_mask)
        x = self.to_out(x)
        
        return x