"""Memory-lean forward for NVIDIA's fast H3 VAE decoder (fast_vae_decoder.py, unchanged).

The reference plain-PyTorch forward builds full-clip temporaries every layer (zero-padded q/k/v windows, padded block
copies for the global tokens, a fp32 [T, H, W, 3072] projection). At 2K x 5 s that peaks around 70 GB, which on a
72 GB card spills into system RAM. This forward computes the same thing one temporal window at a time:

* q / k / v for the whole clip are kept once (bf16); each temporal window (5 latents, shifted by 2 on odd layers) is
  partitioned into its 16x16 spatial windows and attended on its own, with the global tokens of the native blocks it
  overlaps — exactly the reference windows, masks and window order;
* a native block's global tokens attend to that block's real tokens + its own global tokens (the reference pads to
  whole blocks and masks the padding out, which is the same softmax);
* feed-forward, output projection and the final pixel projection run in token / frame chunks.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .fast_vae_decoder import NonCausalViTDecoder, OUT_CH, PATCH, PATCH_T, _apply_qk_norm, _apply_rotary_pos_emb_impl

ROWS = 65536          # tokens per chunk for the per-token linears


def _rows(fn, x, out_dim=None):
    """fn applied to x (..., D) in token chunks."""
    flat = x.reshape(-1, x.shape[-1])
    first = fn(flat[:ROWS])
    out = first.new_empty(flat.shape[0], first.shape[-1])
    out[:first.shape[0]] = first
    for start in range(ROWS, flat.shape[0], ROWS):
        out[start:start + ROWS] = fn(flat[start:start + ROWS])
    return out.view(*x.shape[:-1], first.shape[-1])


class LeanNonCausalViTDecoder(NonCausalViTDecoder):
    def _ff(self, blk, x):
        def ff(t):
            gate, up = blk.ff.w1(blk.norm2(t.float()).to(t.dtype)).chunk(2, dim=-1)
            return blk.ff.w2(F.silu(gate) * up)
        return ff(x) if x.shape[:-1].numel() <= ROWS else _rows(ff, x)

    def _layer(self, li, blk, x, g):
        B, T, H, W, nb, S = self._geom
        dev, tb, nh_ = x.device, self.block, self.heads
        s, st = self._shifts(li)

        def qkv(t):
            h = blk.norm1(t.float()).to(t.dtype)
            return blk.attn.to_qkv(h)

        q, k, v = _rows(qkv, x).view(B, T, H, W, nh_, 3 * self.dh).chunk(3, -1)
        qg, kg, vg = blk.attn.to_qkv(blk.norm1(g.float()).to(g.dtype)).view(B, nb, S, nh_, 3 * self.dh).chunk(3, -1)
        cos, sin = self._rope(T, H, W, s, st, dev)
        cos, sin = cos.view(1, T, H * W, 1, -1), sin.view(1, T, H * W, 1, -1)
        if blk.attn.norm_q is not None:
            qg, kg = _apply_qk_norm(blk.attn.norm_q, qg), _apply_qk_norm(blk.attn.norm_k, kg)
        # q / k: qk-norm + rope per frame (contiguous, one clip-sized copy each); v made contiguous once
        qs = torch.empty(B, T, H, W, nh_, self.dh, device=dev, dtype=x.dtype)
        ks = torch.empty_like(qs)
        for t in range(T):
            qt, kt = q[:, t], k[:, t]
            if blk.attn.norm_q is not None:
                qt, kt = _apply_qk_norm(blk.attn.norm_q, qt), _apply_qk_norm(blk.attn.norm_k, kt)
            rope = (cos[:, t].expand(B, -1, -1, -1), sin[:, t].expand(B, -1, -1, -1))
            qs[:, t] = _apply_rotary_pos_emb_impl(qt.reshape(B, H * W, nh_, self.dh), rope).view(B, H, W, nh_, self.dh)
            ks[:, t] = _apply_rotary_pos_emb_impl(kt.reshape(B, H * W, nh_, self.dh), rope).view(B, H, W, nh_, self.dh)
        vs = v.contiguous()
        del q, k, v

        # ---- window attention, one temporal window at a time
        o = torch.empty(B, T, H, W, nh_ * self.dh, device=dev, dtype=x.dtype)
        nt = (T + st + tb - 1) // tb
        for j in range(nt):
            lo, hi = max(0, j * tb - st), min(T, (j + 1) * tb - st)
            off = lo + st - j * tb                      # first real frame's place inside the window
            qw, vmask, info = self._partition(qs[:, lo:hi], s, off)
            kw, _, _ = self._partition(ks[:, lo:hi], s, off)
            vw, _, _ = self._partition(vs[:, lo:hi], s, off)
            nwin = info[8] * info[9]
            b0, b1 = lo // tb, (hi - 1) // tb
            kgw = kg[:, [b0, b1]].reshape(B, 1, 2 * S, nh_, self.dh).expand(B, nwin, 2 * S, nh_, self.dh)
            vgw = vg[:, [b0, b1]].reshape(B, 1, 2 * S, nh_, self.dh).expand(B, nwin, 2 * S, nh_, self.dh)
            gmask = torch.tensor([True] * S + [b1 != b0] * S, device=dev).expand(nwin, 2 * S)
            Kc = torch.cat([kw, kgw.reshape(B * nwin, 2 * S, nh_, self.dh)], 1).transpose(1, 2)
            Vc = torch.cat([vw, vgw.reshape(B * nwin, 2 * S, nh_, self.dh)], 1).transpose(1, 2)
            kmask = torch.cat([vmask, gmask], 1).repeat(B, 1)[:, None, None, :]
            ow = F.scaled_dot_product_attention(qw.transpose(1, 2), Kc, Vc, attn_mask=kmask).transpose(1, 2)
            o[:, lo:hi] = self._unpartition(ow, info, s, off).reshape(B, hi - lo, H, W, nh_ * self.dh)
            del qw, kw, vw, Kc, Vc, ow

        # ---- global tokens: their native block's real tokens + the block's global tokens
        og = []
        for b in range(nb):
            lo, hi = b * tb, min(T, (b + 1) * tb)
            Kg = torch.cat([ks[:, lo:hi].reshape(B, -1, nh_, self.dh), kg[:, b]], 1).transpose(1, 2)
            Vg = torch.cat([vs[:, lo:hi].reshape(B, -1, nh_, self.dh), vg[:, b]], 1).transpose(1, 2)
            og.append(F.scaled_dot_product_attention(qg[:, b].transpose(1, 2), Kg, Vg).transpose(1, 2).reshape(B, 1, S, -1))
        og = torch.cat(og, 1)
        del qs, ks, vs

        x = x + _rows(lambda t: blk.attn.to_out(t), o) * blk.scale1
        del o
        g = g + blk.attn.to_out(og) * blk.scale1
        x = x + self._ff(blk, x) * blk.scale2
        g = g + self._ff(blk, g) * blk.scale2
        return x, g

    def forward(self, z):
        B, T, _, H, W = z.shape
        tb = self.block
        nb = (T + tb - 1) // tb
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else z.dtype
        raw = z.float() * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2)).permute(0, 2, 3, 4, 1)
        with torch.autocast("cuda", enabled=False):
            x = _rows(lambda t: self.x_embedder(t.float()).to(compute_dtype), raw)
        D = x.shape[-1]
        S = self.register_tokens.shape[1] + 1
        g = torch.cat([self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, D)], 1).to(compute_dtype)
        g = g.unsqueeze(1).expand(B, nb, S, D).contiguous()
        self._geom = (B, T, H, W, nb, S)
        for li, blk in enumerate(self.blocks):
            x, g = self._layer(li, blk, x, g)
        # pixels frame by frame; the first latent of every native block keeps its last frame only
        keep = [PATCH_T - (self.frames_to_trim if t % tb == 0 else 0) for t in range(T)]
        out = torch.empty(B, OUT_CH, sum(keep), H * PATCH, W * PATCH, device=z.device, dtype=torch.float32)
        f = 0
        for t in range(T):
            xt = self.norm_out(x[:, t])
            with torch.autocast("cuda", enabled=False):
                pt = self.proj_out(xt.float())                                   # B, H, W, 3*4*16*16
            pt = pt.view(B, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(0, 3, 4, 1, 5, 2, 6)
            pt = pt.reshape(B, OUT_CH, PATCH_T, H * PATCH, W * PATCH)[:, :, PATCH_T - keep[t]:]
            out[:, :, f:f + keep[t]] = pt
            f += keep[t]
        return out
