import functools
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from unfoldNd import unfoldNd

from tqdm import tqdm


# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, size = 1):
    return val if isinstance(val, tuple) else (val,) * size

def calc_same_padding(kernel_size, dilation = 1):
    return dilation * (kernel_size - 1) // 2

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def sigmoid(t):
    return torch.where(t >= 0, 1 / (1 + torch.exp(-t)), t.exp() / (1 + t.exp()))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def safe_div(numer, denom, eps = 1e-6):
    return numer / (denom + eps)

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    return (t * alpha).softmax(dim = dim)

def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def batch_process(t, fn, chunks = 10, dim = 0):
    chunks = [fn(t_chunk) for t_chunk in t.chunk(chunks, dim = dim)]
    return torch.cat(chunks, dim = dim)

def mult_reduce(arr):
    return functools.reduce(lambda x, y: x * y, arr, 1)

# gradient control

def frac_gradient(t, frac):
    return t * frac + t.detach() * (1 - frac)

# normalizations

class StableLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x / x.amax(dim = -1, keepdim = True).detach()
        return self.norm(x)

class PreNorm(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fn
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class SandwichNorm(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fn
    ):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.postnorm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.prenorm(x)
        x = self.fn(x, **kwargs)
        x = self.postnorm(x)
        return x


# relative positional embedding (rotary)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)



class ShiftVideoTokens(nn.Module):
    def __init__(
        self,
        fn,
        image_size,
        shift_space = True,
        shift_time = False
    ):
        super().__init__()
        self.fn = fn
        self.image_size = image_size

        self.shift_time = shift_time
        self.shift_space = shift_space

    def forward(self, x, **kwargs):

        if not self.shift_time and not self.shift_space:
            return self.fn(x, **kwargs)

        image_size = self.image_size
        img_seq_len = image_size ** 2

        x_bos, x_video = x[:, :1], x[:, 1:]
        n = x_video.shape[1]

        # pad to nearest frame

        padding = img_seq_len - (n % img_seq_len)
        x_video = F.pad(x_video, (0, 0, 0, padding), value = 0.)

        # reshape to video

        x_video = rearrange(x_video, 'b (f h w) d -> b f h w d', h = image_size, w = image_size)

        x_image_h = x_image_w = x_frame = None

        # chunk depending on whether shifting time, space, or both

        if self.shift_space and self.shift_time:
            x_frame, x_image_h, x_image_w, *x_rest = x_video.chunk(5, dim = -1)
        elif self.shift_space:
            x_image_h, x_image_w, *x_rest = x_video.chunk(4, dim = -1)
        elif self.shift_time:
            x_frame, *x_rest = x_video.chunk(3, dim = -1)

        # shifts

        if self.shift_space:
            x_image_h = F.pad(x_image_h, (0, 0, 0, 0, 1, -1))
            x_image_w = F.pad(x_image_w, (0, 0, 1, -1))

        if self.shift_time:
            x_frame = F.pad(x_frame, (0, 0, 0, 0, 0, 0, 1, -1))

        # concat

        x_shifted = [x_frame, x_image_h, x_image_w, *x_rest]
        x_shifted = list(filter(exists, x_shifted))

        x_video = torch.cat(x_shifted, dim = -1)

        # merge text and image sequence back together

        x_video = rearrange(x_video, 'b f h w d -> b (f h w) d')
        x_video = x_video[:, :n]

        x = torch.cat((x_bos, x_video), dim = 1)
        return self.fn(x, **kwargs)
    


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)



class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.,
        chunk_size = None,  # chunk size to process feedforward, along sequence length, from Reformer paper. None means do not chunk
    ):
        super().__init__()
        inner_dim = (dim * mult * 2) // 3
        self.chunk_size = chunk_size

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        if not exists(self.chunk_size):
            return self.net(x)

        x_chunks = x.split(self.chunk_size, dim = -2)
        out_chunks = [self.net(c) for c in x_chunks]
        return torch.cat(out_chunks, dim = -2)




# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5

        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None,
        rotary_pos_emb = None
    ):
        b, h, device = x.shape[0], self.heads, x.device

        has_context = exists(context)
        kv_input = context if has_context else x

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # add rotary positional embedding, if exists

        if not has_context and exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        # add null key / values, needed for condition dropout

        null_k = repeat(self.null_k, 'h 1 d -> b h 1 d', b = b)
        null_v = repeat(self.null_v, 'h 1 d -> b h 1 d', b = b)

        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        # scale

        q = q * self.scale

        # similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        mask_value = -torch.finfo(x.dtype).max

        key_mask = mask if not has_context else context_mask

        if exists(key_mask):
            key_mask = F.pad(key_mask, (1, 0), value = True) # always pay attention to null key / value
            key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~key_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(j - i + 1)
            sim = sim.masked_fill(mask, mask_value)

        # attention

        attn = stable_softmax(sim, dim = -1)
        attn = self.talking_heads(attn)
        attn = self.dropout(attn)

        # aggregate, merge, and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Sparse3DNA(nn.Module):
    def __init__(
        self,
        dim,
        video_shape,
        kernel_size = 3,
        dilation = 1,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False,
        query_num_frames_chunk = None,
        rel_pos_bias = False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dilation = cast_tuple(dilation, size = 3)

        self.kernel_size = cast_tuple(kernel_size, size = 3)
        assert all(map(lambda n: n % 2 == 1, self.kernel_size)), 'kernel size must be odd'

        self.kernel_numel = mult_reduce(self.kernel_size)

        # relative positional bias per head, if needed

        self.rel_pos_bias = AxialPositionalEmbedding(heads, shape = self.kernel_size) if rel_pos_bias else None

        # calculate padding

        self.padding_frame = calc_same_padding(self.kernel_size[0], self.dilation[0])
        self.padding_height = calc_same_padding(self.kernel_size[1], self.dilation[1])
        self.padding_width = calc_same_padding(self.kernel_size[2], self.dilation[2])

        # use separate padding for causal convolution vs non-causal

        if causal:
            self.video_padding = (self.padding_width * 2, 0, self.padding_height * 2, 0, self.padding_frame * 2, 0)
        else:
            self.video_padding = (self.padding_width, self.padding_width, self.padding_height, self.padding_height, self.padding_frame, self.padding_frame)

        # save video shape and calculate max number of tokens

        self.video_shape = video_shape
        max_frames, fmap_size, _ = video_shape
        max_num_tokens = torch.empty(video_shape).numel()
        self.max_num_tokens = max_num_tokens

        # how many query tokens to process at once to limit peak memory usage, by multiple of frame tokens (fmap_size ** 2)

        self.query_num_frames_chunk = default(query_num_frames_chunk, max_frames)

        # precalculate causal mask

        ones = torch.ones((max_num_tokens,))
        ones = rearrange(ones, '(f h w) -> 1 1 f h w', f = max_frames, h = fmap_size, w = fmap_size)
        ones = F.pad(ones, self.video_padding, value = 0.)
        ones = unfoldNd(ones, kernel_size = self.kernel_size, dilation = self.dilation)
        ones = rearrange(ones, '1 k n -> n k')

        # mask out padding

        padding_mask = ones == 0.

        # bos tokens never get masked out

        mask = F.pad(padding_mask, (1, 0), value = False)
        self.register_buffer('mask', mask)

    def forward(self, x, **kwargs):
        b, n, _, h, device = *x.shape, self.heads, x.device

        # more variables

        dilation = self.dilation
        kernel_size = self.kernel_size
        video_padding = self.video_padding
        fmap_size = self.video_shape[1]

        bos_only = n == 1
        tokens_per_frame = fmap_size ** 2

        padding = padding_to_multiple_of(n - 1, tokens_per_frame)
        num_frames = (n + padding) // tokens_per_frame

        # pad for last token in video

        padded_x = F.pad(x, (0, 0, 0, padding), value = 0.) if padding > 0 else x

        # derive queries / keys / values

        q, k, v = (self.to_q(x), *self.to_kv(padded_x).chunk(2, dim = -1))

        # early return if <bos>

        if bos_only:
            return self.to_out(v)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # scale queries

        q = q * self.scale

        # take care of bos

        q = q[:, 1:]
        bos_value = v[:, :1]

        # compute keys and values

        (k_bos, k), (v_bos, v) = map(lambda t: (t[:, :1], t[:, 1:]), (k, v))

        # reshape keys and values to video and add appropriate padding along all dimensions (frames, height, width)

        k, v = map(lambda t: rearrange(t, 'b (f h w) d -> b d f h w', f  = num_frames, h = fmap_size), (k, v))
        k, v = map(lambda t: F.pad(t, video_padding), (k, v))

        # axial relative pos bias

        rel_pos_bias = None

        if exists(self.rel_pos_bias):
            rel_pos_bias = rearrange(self.rel_pos_bias(), 'j h -> h 1 j')
            rel_pos_bias = F.pad(rel_pos_bias, (1, 0), value = 0.)

        # put the attention processing code in a function
        # to allow for processing queries in chunks of frames

        out = []

        def attend(q, k, v, mask, k_bos, v_bos, kernel_size):
            chunk_length = q.shape[1]

            k, v = map(lambda t: unfoldNd(t, kernel_size = kernel_size, dilation = dilation), (k, v))
            k, v = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = self.kernel_numel), (k, v))
            k, v = map(lambda t: t[:, :chunk_length], (k, v))

            # append bos keys and values

            k_bos, v_bos = map(lambda t: repeat(t, 'b 1 d -> b n 1 d', n = k.shape[1]), (k_bos, v_bos))
            k = torch.cat((k_bos, k), dim = 2)
            v = torch.cat((v_bos, v), dim = 2)

            # calculate sim

            sim = einsum('b i d, b i j d -> b i j', q, k)

            # add rel pos bias, if needed

            if exists(rel_pos_bias):
                sim = sim + rel_pos_bias

            # causal mask

            if exists(mask):
                mask_value = -torch.finfo(sim.dtype).max
                mask = rearrange(mask, 'i j -> 1 i j')
                sim = sim.masked_fill(mask, mask_value)

            # attention

            attn = stable_softmax(sim, dim = -1)

            attn = rearrange(attn, '(b h) ... -> b h ...', h = h)
            attn = self.talking_heads(attn)
            attn = rearrange(attn, 'b h ... -> (b h) ...')

            attn = self.dropout(attn)

            # aggregate values

            return einsum('b i j, b i j d -> b i d', attn, v)

        # process queries in chunks

        frames_per_chunk = min(self.query_num_frames_chunk, num_frames)
        chunk_size = frames_per_chunk * tokens_per_frame

        q_chunks = q.split(chunk_size, dim = 1)

        mask = self.mask[:(n - 1)]
        mask_chunks = mask.split(chunk_size, dim = 0)

        for ind, (q_chunk, mask_chunk) in enumerate(zip(q_chunks, mask_chunks)):
            q_chunk = q_chunks[ind]
            mask_chunk = mask_chunks[ind]

            # slice the keys and values to the appropriate frames, accounting for padding along frames dimension

            kv_start_pos = ind * frames_per_chunk
            kv_end_pos = kv_start_pos + (ind + frames_per_chunk + self.padding_frame * 2)
            kv_frame_range = slice(kv_start_pos, kv_end_pos)

            k_slice, v_slice = map(lambda t: t[:, :, kv_frame_range], (k, v))

            # calculate output chunk

            out_chunk = attend(
                q = q_chunk,
                k = k_slice,
                v = v_slice,
                mask = mask_chunk,
                k_bos = k_bos,
                v_bos = v_bos,
                kernel_size = kernel_size,
            )

            out.append(out_chunk)

        # combine all chunks

        out = torch.cat(out, dim = 1)

        # append bos value

        out = torch.cat((bos_value, out), dim = 1)  # bos will always adopt its own value, since it pays attention only to itself

        # merge heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)



class SparseCross2DNA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        kernel_size = 3,
        dilation = 1,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

        self.talking_heads = nn.Conv3d(heads, heads, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # handle variables for 2d unfold

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = calc_same_padding(kernel_size, dilation)

    def forward(
        self,
        x,
        *,
        context,
        context_mask = None,
        **kwargs
    ):
        b, n, h, device = x.shape[0], x.shape[1], self.heads, x.device

        fmap_size, kernel_size, dilation, padding = self.image_size, self.kernel_size, self.dilation, self.padding

        context_len = context.shape[-2]
        tokens_per_frame = fmap_size * fmap_size
        kernel_numel = kernel_size * kernel_size

        # always have context mask avaiable

        if not exists(context_mask):
            context_mask = torch.ones((b, context_len), dtype = torch.bool, device = device)

        mask_value = -torch.finfo(x.dtype).max

        # derive queries, keys, values

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # scale

        q = q * self.scale

        # handle bos

        q_bos, q = q[:, :, 0], q[:, :, 1:]

        null_k_for_bos = repeat(self.null_k, 'h 1 d -> b h 1 d', b = b)
        null_v_for_bos = repeat(self.null_v, 'h 1 d -> b h 1 d', b = b)

        k_for_bos = torch.cat((null_k_for_bos, k), dim = -2)
        v_for_bos = torch.cat((null_v_for_bos, v), dim = -2)

        sim_bos = einsum('b h d, b h j d -> b h j', q_bos, k_for_bos)

        bos_context_mask = rearrange(context_mask, 'b j -> b 1 j')
        bos_context_mask = F.pad(bos_context_mask, (1, 0), value = True)
        sim_bos = sim_bos.masked_fill(~bos_context_mask, mask_value)

        attn_bos = stable_softmax(sim_bos, dim = -1)
        out_bos = einsum('b h j, b h j d -> b h d', attn_bos, v_for_bos)
        out_bos = rearrange(out_bos, 'b h d -> b 1 (h d)')

        # early return if only bos token

        if n == 1:
            return self.to_out(out_bos)

        # reshape key / values to be unfolded

        k, v = map(lambda t: rearrange(t, 'b h (f x y) d -> (b h f) d x y', x = fmap_size, y = fmap_size), (k, v))
        k, v = map(lambda t: F.unfold(t, kernel_size = kernel_size, dilation = dilation, padding = padding), (k, v))
        k, v = map(lambda t: rearrange(t, '(b h f) (d j) i -> b h i (f j) d', b = b, h = h, j = kernel_numel), (k, v))

        # add null key / values, needed for condition dropout

        null_k = repeat(self.null_k, 'h 1 d -> b h i 1 d', b = b, i = tokens_per_frame)
        null_v = repeat(self.null_v, 'h 1 d -> b h i 1 d', b = b, i = tokens_per_frame)

        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        # pad queries to nearest frame

        q_padding = padding_to_multiple_of(q.shape[-2], tokens_per_frame)
        q = F.pad(q, (0, 0, 0, q_padding), value = 0.)

        # similarity

        q = rearrange(q, 'b h (f i) d -> b h f i d', i = tokens_per_frame)

        sim = einsum('b h f i d, b h i j d -> b h f i j', q, k)

        # masking

        context_mask = rearrange(context_mask, 'b (f x y) -> (b f) 1 x y', x = fmap_size, y = fmap_size)
        context_mask = F.unfold(context_mask.float(), kernel_size = kernel_size, dilation = dilation, padding = padding)
        context_mask = context_mask == 1.
        context_mask = rearrange(context_mask, '(b f) j i -> b 1 1 i (f j)', b = b, j = kernel_numel)
        context_mask = F.pad(context_mask, (1, 0), value = True) # always pay attention to null key / value

        sim = sim.masked_fill(~context_mask, mask_value)

        # attention

        attn = stable_softmax(sim, dim = -1)
        attn = self.talking_heads(attn)
        attn = self.dropout(attn)

        # aggregate, merge, and combine heads

        out = einsum('b h f i j, b h i j d -> b h f i d', attn, v)
        out = rearrange(out, 'b h f n d -> b (f n) (h d)')

        # add output for bos back

        out = torch.cat((out_bos, out), dim = 1)

        return self.to_out(out[:, :n])
    


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        cross_attend = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        cross_2dna_attn = False,
        cross_2dna_image_size = None,
        cross_2dna_kernel_size = 3,
        cross_2dna_dilations = (1,),
        sparse_3dna_attn = False,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_video_shape = None,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        sparse_3dna_rel_pos_bias = False,
        shift_video_tokens = False,
        rotary_pos_emb = False
    ):
        super().__init__()
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'
        assert not (cross_2dna_attn and not exists(cross_2dna_image_size)), 'cross_2dna_image_size must be defined'

        self.layers = MList([])

        for ind in range(depth):
            if sparse_3dna_attn:
                dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]

                self_attn = Sparse3DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    kernel_size = sparse_3dna_kernel_size,
                    dilation = dilation,
                    video_shape = sparse_3dna_video_shape,
                    query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
                    rel_pos_bias = sparse_3dna_rel_pos_bias,
                )
            else:
                self_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    dropout = attn_dropout
                )

            cross_attn = None

            if cross_attend:
                if cross_2dna_attn:
                    dilation = cross_2dna_dilations[ind % len(cross_2dna_dilations)]

                    cross_attn = SparseCross2DNA(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        dropout = attn_dropout,
                        image_size = cross_2dna_image_size,
                        kernel_size = cross_2dna_kernel_size,
                        dilation = dilation
                    )

                else:
                    cross_attn = Attention(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        dropout = attn_dropout
                    )

            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            if sparse_3dna_attn and shift_video_tokens:
                fmap_size = sparse_3dna_video_shape[-1]
                self_attn = ShiftVideoTokens(self_attn, image_size = fmap_size)
                ff        = ShiftVideoTokens(ff, image_size = fmap_size)

            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = self_attn),
                SandwichNorm(dim = dim, fn = cross_attn) if cross_attend else None,
                SandwichNorm(dim = dim, fn = ff)
            ]))

        self.norm = StableLayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None
    ):
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask = mask) + x

            if exists(cross_attn):
                x = cross_attn(x, context = context, mask = mask, context_mask = context_mask) + x

            x = ff(x) + x

        return self.norm(x)



class Embedding(nn.Module):
    def __init__(self, *shape, frac_gradient = 1.):
        super().__init__()
        self.frac_gradient = frac_gradient
        self.embed = nn.Embedding(*shape)

    def forward(self, x):
        x = self.embed(x)

        if self.training and self.frac_gradient < 1:
            x = frac_gradient(x, self.frac_gradient)

        return x



# positional embedding

class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        *,
        shape
    ):
        super().__init__()
        shape = tuple(filter(lambda t: t > 1, shape))

        self.dim = dim
        self.shape = shape
        self.num_axials = len(shape)

        for axial_ind, axial_len in enumerate(shape):
            axial_pos = nn.Parameter(torch.randn(axial_len, dim))
            setattr(self, f'axial{axial_ind + 1}', axial_pos)

    def forward(self, *, flatten = True):
        positions = None

        for axial_ind in range(self.num_axials):
            axial_pos = getattr(self, f'axial{axial_ind + 1}')

            if not exists(positions):
                positions = axial_pos
                continue

            positions = rearrange(positions, '... d -> ... 1 d')
            positions = positions + axial_pos

        if flatten:
            positions = rearrange(positions, '... d -> (...) d')

        return positions



# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs



class NUWA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae = None,
        image_size = None,
        max_video_frames = 5,
        text_num_tokens = 49408,
        text_max_seq_len = 256,
        text_enc_depth = 6,
        text_enc_dim_head = 64,
        text_enc_heads = 8,
        text_rotary_pos_emb = True,
        enc_reversible = False,
        dec_depth = 6,
        dec_dim_head = 64,
        dec_heads = 8,
        dec_reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        embed_gradient_frac = 0.2,
        shift_video_tokens = True,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilation = 1,
        sparse_3dna_rel_pos_bias = False
    ):
        super().__init__()
        assert exists(vae) ^ exists(image_size), 'either VAE or image size must be specified'

        self.vae = None
        if exists(vae):
            self.vae = vae.copy_for_eval()
            image_size = vae.image_size

        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = Embedding(text_num_tokens, dim, frac_gradient = embed_gradient_frac)

        # positional embedding for text

        self.text_abs_pos_emb = Embedding(text_max_seq_len, dim)  if not text_rotary_pos_emb else None
        self.text_rotary_pos_emb = RotaryEmbedding(dim = min(32, text_enc_dim_head)) if text_rotary_pos_emb else None

        enc_transformer_klass = Transformer

        self.text_transformer = enc_transformer_klass(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            rotary_pos_emb = text_rotary_pos_emb
        )

        self.video_bos = nn.Parameter(torch.randn(dim))
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        fmap_size = image_size // (2 ** vae_num_layers)

        self.video_fmap_size = fmap_size
        self.max_video_frames = max_video_frames
        video_shape = (max_video_frames, fmap_size, fmap_size)

        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # cycle dilation for sparse 3d-nearby attention

        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        dec_transformer_klass = Transformer

        self.video_transformer = dec_transformer_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            causal = True,
            cross_attend = True,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_attn = True,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_rel_pos_bias = sparse_3dna_rel_pos_bias
        )

        self.to_logits = nn.Linear(dim, num_image_tokens, bias = False)

    def embed_text(self, text, mask = None):
        batch, seq_len, device = *text.shape, text.device
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        tokens = self.text_embedding(text)

        if exists(self.text_abs_pos_emb):
            pos_emb = self.text_abs_pos_emb(torch.arange(seq_len, device = device))
            tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.text_rotary_pos_emb):
            rotary_pos_emb = self.text_rotary_pos_emb(seq_len, device = device)

        return self.text_transformer(
            tokens,
            mask = mask,
            rotary_pos_emb = rotary_pos_emb
        )


    def forward(
        self,
        *,
        text,
        video = None,
        return_loss = False,
        cond_dropout_prob = 0.2
    ):
        batch, seq_len, frames, device = *text.shape, video.shape[1], text.device

        text_mask = text != 0
        text_embeds = self.embed_text(text, mask = text_mask)

        if video.dtype == torch.long:
            frame_indices = video
        else:
            assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'
            assert exists(self.vae), 'VAE must be passed in if you wish for video to be encoded to ids automatically'
            frame_indices = self.vae.get_video_indices(video)

        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        frame_indices_input = frame_indices[:, :-1] if return_loss else frame_indices

        frame_embeddings = self.image_embedding(frame_indices_input)
        frame_embeddings = self.video_pos_emb()[:-1] + frame_embeddings

        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

        if self.training and cond_dropout_prob > 0:
            # dropout condition randomly
            # presented in https://openreview.net/forum?id=qw8AKxfYbI
            uncond_mask = prob_mask_like((batch,), cond_dropout_prob, device = device)
            text_mask *= rearrange(~uncond_mask, 'b -> b 1')

        frame_embeddings = self.video_transformer(
            frame_embeddings,
            context = text_embeds,
            context_mask = text_mask
        )

        logits = self.to_logits(frame_embeddings)

        if not return_loss:
            return logits

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), frame_indices)
        return loss