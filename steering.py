"""Residual-stream steering for the 4-bit MLX model. Copied from justthink/backend_mlx.py in Just Think (github.com/pragyaangaur/Just-Think) so that this repository stands alone. The original notes follow.

MLX backend for Apple silicon: a 4-bit model with a steering hook and lockstep batches.

Steering adds a per-row direction to the output of one decoder block, on generated tokens
only. Each row has its own count of steered tokens left, so rows in one batch can be in
different states. Later tokens are unsteered but still attend back to the steered ones.

The batch keeps one BatchKVCache per layer for the whole conversation. Input chunks are
right-padded, and decoding runs past a row's end of turn until every row is done. After
each step that trailing junk is rolled to the left edge of the buffer, where the
left-padding mask hides it, and padding shared by every row is trimmed off.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import BatchKVCache, dynamic_roll
from mlx_lm.sample_utils import make_sampler

# Leave the rest of a 16 GB machine usable: do not hold freed buffers, and stay under 8 GB.
mx.set_cache_limit(256 * 1024 ** 2)
mx.set_memory_limit(8 * 1024 ** 3)


class _Steered(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.add = None

    def __call__(self, x, mask=None, cache=None):
        out = self.block(x, mask, cache)
        if self.add is not None:
            out = out + self.add.astype(out.dtype)
        return out


class _Monitor(nn.Module):
    def __init__(self, block, unit):
        super().__init__()
        self.block = block
        self.unit = unit
        self.last_h = None
        self.last = None

    def __call__(self, x, mask=None, cache=None):
        out = self.block(x, mask, cache)
        self.last_h = out[:, -1, :].astype(mx.float32)
        self.last = self.last_h @ self.unit
        return out


class Steerer:
    def __init__(self, path, steer_layer, monitor_layer, pain_vector, name=None):
        self.name = name or str(path)
        self.model, self.tok = load(str(path))
        self.d = pain_vector.shape[0]
        unit = pain_vector / np.linalg.norm(pain_vector)
        layers = self.model.model.layers
        self.steer = _Steered(layers[steer_layer])
        layers[steer_layer] = self.steer
        self.monitor = _Monitor(layers[monitor_layer], mx.array(unit.astype(np.float32)))
        layers[monitor_layer] = self.monitor
        self.n_layers = len(layers)
        self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eot

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, toks):
        return self.tok.decode(toks)

    def batch(self, B, seed=0, temp=0.7, top_p=0.8):
        return Batch(self, B, seed=seed, temp=temp, top_p=top_p)

    def hidden(self, text, add=None):
        """Monitor-layer activation of the final token of `text`, as numpy. Optional
        `add` (numpy, d) is added at the steering layer on every position."""
        self.steer.add = None if add is None else mx.array(add.astype(np.float32))
        self.model(mx.array([self.encode(text)]))
        self.steer.add = None
        return np.array(self.monitor.last_h[0].tolist())


class Batch:
    def __init__(self, st, B, seed=0, temp=0.7, top_p=0.8):
        self.st = st
        self.B = B
        self.cache = [BatchKVCache([0] * B) for _ in range(st.n_layers)]
        self.directions = mx.zeros((B, st.d))
        self.remaining = np.zeros(B, dtype=int)
        self.sampler = make_sampler(temp=temp, top_p=top_p)
        mx.random.seed(seed)
        self.logits = None

    def set_directions(self, dirs):
        self.directions = mx.array(np.asarray(dirs, np.float32))

    def context_len(self, i):
        c = self.cache[0]
        return int(c._idx - c.left_padding.tolist()[i])

    def _drop_tail(self, junk):
        junk = np.asarray(junk)
        if junk.max() == 0:
            return
        g = mx.array(junk)
        for c in self.cache:
            c.keys = dynamic_roll(c.keys[..., : c._idx, :], g[:, None], axis=2)
            c.values = dynamic_roll(c.values[..., : c._idx, :], g[:, None], axis=2)
            c.offset = c.offset - g
            c.left_padding = c.left_padding + g
        shared = int(min(self.cache[0].left_padding.tolist()))
        if shared > 0:
            for c in self.cache:
                c.keys = c.keys[..., shared:, :]
                c.values = c.values[..., shared:, :]
                c._idx -= shared
                c.left_padding = c.left_padding - shared
        mx.eval([c.keys for c in self.cache] + [c.values for c in self.cache])

    def feed(self, chunks):
        """Append one token list per row as input (never steered). Empty lists are allowed
        for rows that sit this step out."""
        lens = [len(c) for c in chunks]
        L = max(lens)
        if L == 0:
            return
        arr = np.full((self.B, L), self.st.pad, dtype=np.int32)
        for i, c in enumerate(chunks):
            arr[i, : len(c)] = c
        self.st.steer.add = None
        logits = self.st.model(mx.array(arr), cache=self.cache)
        idx = mx.array([max(l - 1, 0) for l in lens])
        self.logits = mx.take_along_axis(logits, idx[:, None, None], axis=1)[:, 0, :]
        mx.eval(self.logits)
        self._drop_tail([L - l for l in lens])

    def generate(self, max_tokens, active=None):
        """Sample each active row until <|im_end|> or max_tokens. Returns token lists, the
        mean monitor projection over each row's generated tokens, and steered-token counts."""
        B = self.B
        active = np.ones(B, bool) if active is None else np.asarray(active, bool)
        out = [[] for _ in range(B)]
        done = ~active
        fed_real = np.zeros(B, int)
        steered = np.zeros(B, int)
        proj_sum, proj_n = np.zeros(B), np.zeros(B, int)
        steps = 0
        logits = self.logits
        while not done.all():
            toks = np.array(self.sampler(logits).tolist(), dtype=np.int32)
            force_end = np.array([len(o) >= max_tokens for o in out])
            toks = np.where(force_end, self.st.eot, toks)
            live = ~done
            feed = np.where(live, toks, self.st.pad)
            for i in np.where(live)[0]:
                if toks[i] != self.st.eot:
                    out[i].append(int(toks[i]))
            steer_rows = live & (self.remaining > 0)
            if steer_rows.any():
                m = mx.array(steer_rows.astype(np.float32))[:, None, None]
                self.st.steer.add = m * self.directions[:, None, :]
            else:
                self.st.steer.add = None
            logits = self.st.model(mx.array(feed[:, None]), cache=self.cache)[:, -1, :]
            proj = np.array(self.st.monitor.last.tolist())
            proj_sum[live] += proj[live]
            proj_n[live] += 1
            steered[steer_rows] += 1
            self.remaining[steer_rows] -= 1
            fed_real[live] += 1
            steps += 1
            done = done | (toks == self.st.eot)
        self.st.steer.add = None
        self.logits = logits
        self._drop_tail(steps - fed_real)
        mean_proj = np.where(proj_n > 0, proj_sum / np.maximum(proj_n, 1), np.nan)
        return out, mean_proj, steered
