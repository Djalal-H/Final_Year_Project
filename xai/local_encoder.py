"""Local encoder classes for attention extraction.

This module contains custom encoder implementations that properly handle
attention weight extraction during training. These classes are based on the
offline attention extraction notebook and ensure correct shape handling,
especially for roadgraph data.
"""

from functools import partial
from typing import Callable

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents import datatypes
from vmax.agents.networks import encoders


def default(val, d):
    """Return val if val is not None, else d."""
    return val if val is not None else d


class LocalAttentionLayer(nn.Module):
    """Local version of AttentionLayer with return_attention_weights support."""
    heads: int = 8
    head_features: int = 64
    dropout: float = 0.0
    return_attention_weights: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, context=None, mask_k=None, mask_q=None, deterministic: bool = False):
        h = self.heads
        dim = self.head_features * h

        q = nn.Dense(dim, use_bias=False)(x)
        k = nn.Dense(dim, use_bias=False)(default(context, x))
        v = nn.Dense(dim, use_bias=False)(default(context, x))

        q, k, v = map(lambda arr: einops.rearrange(arr, "b n (h d) -> b n h d", h=h), (q, k, v))
        sim = jnp.einsum("b i h d, b j h d -> b i j h", q, k) * self.head_features**-0.5

        if mask_k is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_k[:, None, :, None], sim, big_neg)
        if mask_q is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_q[:, :, None, None], sim, big_neg)

        attn = nn.softmax(sim, axis=-2)
        attn_weights_for_analysis = attn if self.return_attention_weights else None
        
        out = jnp.einsum("b i j h, b j h d -> b i h d", attn, v)
        out = einops.rearrange(out, "b n h d -> b n (h d)", h=h)

        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        if self.return_attention_weights:
            return out, attn_weights_for_analysis
        return out


class LocalWayformerAttention(nn.Module):
    """Local Wayformer attention module with attention weight extraction."""
    depth: int = 2
    num_latents: int = 32
    num_heads: int = 2
    head_features: int = 16
    ff_mult: int = 1
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    return_attention_weights: bool = False

    @nn.compact
    def __call__(self, x, mask=None):
        bs, dim = x.shape[0], x.shape[-1]
        latents = self.param("latents", init.normal(), (self.num_latents, dim * self.ff_mult))
        latent = einops.repeat(latents, "n d -> b n d", b=bs)
        x = einops.rearrange(x, "b n ... -> b n (...)")

        attention_weights = {} if self.return_attention_weights else None

        # Use LOCAL attention layer instead of encoders.AttentionLayer
        attn = partial(
            LocalAttentionLayer,
            heads=self.num_heads,
            head_features=self.head_features,
            dropout=self.attn_dropout,
            return_attention_weights=self.return_attention_weights
        )
        ff = partial(encoders.FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)
        
        # Cross-attention (attn_0)
        rz = encoders.ReZero(name="rezero_0")
        if self.return_attention_weights:
            attn_out, attn_w = attn(name="attn_0")(latent, x, mask_k=mask)
            latent += rz(attn_out)
            attention_weights['cross_attn_0'] = attn_w
        else:
            latent += rz(attn(name="attn_0")(latent, x, mask_k=mask))
        latent += rz(ff(name="ff_0")(latent))

        # Self-attention layers
        for i in range(1, self.depth):
            rz = encoders.ReZero(name=f"rezero_{i}")
            if self.return_attention_weights:
                attn_out, attn_w = attn(name=f"attn_{i}")(latent)
                latent += rz(attn_out)
                attention_weights[f'self_attn_{i}'] = attn_w
            else:
                latent += rz(attn(name=f"attn_{i}")(latent))
            latent += rz(ff(name=f"ff_{i}")(latent))

        if self.return_attention_weights:
            return latent, attention_weights
        return latent


class LocalWayformerEncoder(nn.Module):
    """Local Wayformer encoder with proper shape handling for attention extraction.
    
    This encoder handles roadgraph features specially - they already have the correct
    batch dimension and should not get an extra dimension added, unlike other features.
    """
    unflatten_fn: Callable = lambda x: x
    embedding_layer_sizes: tuple = (256, 256)
    embedding_activation: datatypes.ActivationFn = nn.relu
    attention_depth: int = 2
    dk: int = 64
    num_latents: int = 64
    latent_num_heads: int = 4
    latent_head_features: int = 64
    ff_mult: int = 2
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    fusion_type: str = "late"
    return_attention_weights: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array):
        # Add batch dimension if missing
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)
        
        features, masks = self.unflatten_fn(obs)
        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = features
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        # Ensure all features have batch dimension
        # CRITICAL: roadgraph already has correct shape (batch, num_points, features) - don't add extra dim
        def ensure_batch_dim(x):
            if x.ndim == 2:
                return jnp.expand_dims(x, axis=0)
            return x
        
        sdc_traj_features = ensure_batch_dim(sdc_traj_features)
        other_traj_features = ensure_batch_dim(other_traj_features)
        # rg_features already has correct shape (batch, num_points, features) - don't add extra dim
        tl_features = ensure_batch_dim(tl_features)
        gps_path_features = ensure_batch_dim(gps_path_features)
        sdc_traj_valid_mask = ensure_batch_dim(sdc_traj_valid_mask)
        other_traj_valid_mask = ensure_batch_dim(other_traj_valid_mask)
        # rg_valid_mask already has correct shape (batch, num_points) - don't add extra dim
        tl_valid_mask = ensure_batch_dim(tl_valid_mask)

        num_objects, timestep_agent = other_traj_features.shape[-3:-1]
        num_roadgraph = rg_features.shape[-2]
        target_len = gps_path_features.shape[-2]
        num_light, timestep_tl = tl_features.shape[-3:-1]

        # Embeddings
        sdc_traj_encoding = encoders.build_mlp_embedding(sdc_traj_features, self.dk, self.embedding_layer_sizes, self.embedding_activation, "sdc_traj_enc")
        other_traj_encoding = encoders.build_mlp_embedding(other_traj_features, self.dk, self.embedding_layer_sizes, self.embedding_activation, "other_traj_enc")
        rg_encoding = encoders.build_mlp_embedding(rg_features, self.dk, self.embedding_layer_sizes, self.embedding_activation, "rg_enc")
        tl_encoding = encoders.build_mlp_embedding(tl_features, self.dk, self.embedding_layer_sizes, self.embedding_activation, "tl_enc")
        gps_path_encoding = encoders.build_mlp_embedding(gps_path_features, self.dk, self.embedding_layer_sizes, self.embedding_activation, "gps_path_enc")

        # PE and Temporal Encoding
        sdc_traj_encoding += jnp.expand_dims(self.param("sdc_traj_pe", init.normal(), (1, timestep_agent, self.dk)), 0)
        other_traj_encoding += jnp.expand_dims(self.param("other_traj_pe", init.normal(), (num_objects, timestep_agent, self.dk)), 0)
        rg_encoding += self.param("rg_pe", init.normal(), (num_roadgraph, self.dk))[None, :, :]
        tl_encoding += jnp.expand_dims(self.param("tj_pe", init.normal(), (num_light, timestep_tl, self.dk)), 0)
        gps_path_encoding += jnp.expand_dims(self.param("gps_path_pe", init.normal(), (target_len, self.dk)), 0)

        temp_pe_agents = self.param("temp_pe_agents", init.normal(), (timestep_agent,))
        temp_pe_tl = self.param("temp_pe_tl", init.normal(), (timestep_tl,))
        sdc_traj_encoding += temp_pe_agents[None, None, :, None]
        other_traj_encoding += temp_pe_agents[None, None, :, None]
        tl_encoding += temp_pe_tl[None, None, :, None]

        # Reshaping
        sdc_traj_encoding = einops.rearrange(sdc_traj_encoding, "b n t d -> b (n t) d")
        other_traj_encoding = einops.rearrange(other_traj_encoding, "b n t d -> b (n t) d")
        tl_encoding = einops.rearrange(tl_encoding, "b n t d -> b (n t) d")
        
        # Roadgraph has no time dimension, already in shape (b, n, d) - no rearrange needed
        # rg_encoding stays as is
        
        sdc_traj_valid_mask = einops.rearrange(sdc_traj_valid_mask, "b n t -> b (n t)")
        other_traj_valid_mask = einops.rearrange(other_traj_valid_mask, "b n t -> b (n t)")
        tl_valid_mask = einops.rearrange(tl_valid_mask, "b n t -> b (n t)")
        # rg_valid_mask stays as is
        
        all_attention_weights = {} if self.return_attention_weights else None

        self_attn = partial(
            LocalWayformerAttention,
            num_latents=self.num_latents,
            num_heads=self.latent_num_heads,
            head_features=self.latent_head_features,
            ff_mult=self.ff_mult,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            return_attention_weights=self.return_attention_weights
        )

        def call_attn(attn_module, embeddings, mask, prefix):
            if self.return_attention_weights:
                out, attn_w = attn_module()(embeddings, mask)
                for k, v in attn_w.items():
                    all_attention_weights[f'{prefix}/{k}'] = v
                return out
            else:
                return attn_module()(embeddings, mask)

        # Late fusion
        output_sdc_traj = call_attn(partial(self_attn, depth=self.attention_depth, name="sdc_traj_attention"), sdc_traj_encoding, sdc_traj_valid_mask, "sdc_traj")
        output_other_traj = call_attn(partial(self_attn, depth=self.attention_depth, name="other_traj_attention"), other_traj_encoding, other_traj_valid_mask, "other_traj")
        output_rg = call_attn(partial(self_attn, depth=self.attention_depth, name="rg_attention"), rg_encoding, rg_valid_mask, "roadgraph")
        output_tl = call_attn(partial(self_attn, depth=self.attention_depth, name="tl_attention"), tl_encoding, tl_valid_mask, "traffic_lights")
        output_gps_path = call_attn(partial(self_attn, depth=self.attention_depth, name="gps_path_attention"), gps_path_encoding, None, "gps_path")

        output = jnp.concatenate([output_sdc_traj, output_other_traj, output_rg, output_tl, output_gps_path], axis=-2)
        output = output.mean(axis=1)

        if self.return_attention_weights:
            return output, all_attention_weights
        return output
