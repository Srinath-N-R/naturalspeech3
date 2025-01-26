import math
import torch
from torch import nn
import torch.nn.functional as F
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder


def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)


def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)


def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)


class ConvolutionalFeedForward(nn.Module):
    """
    A feed-forward block that uses 1D convolution for local context,
    followed by a pointwise feed-forward.
    """
    def __init__(self, embed_dim, ff_dim, kernel_size=9, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=ff_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x shape: (B, S, E)
        Transform to (B, E, S) to apply 1D conv over the sequence dimension.
        """
        x = x.transpose(1, 2)            # (B, E, S)
        x = self.conv1(x)               # (B, ff_dim, S)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)               # (B, embed_dim, S)
        x = x.transpose(1, 2)           # (B, S, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer block with self-attention and convolutional feedforward.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, kernel_size=9, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.feedforward = ConvolutionalFeedForward(embed_dim, ff_dim, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.layernorm1(x)

        # Feedforward (with conv)
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.layernorm2(x)
        return x


class PhonemeEncoder(nn.Module):
    """
    Phoneme encoder with 6 Transformer layers.
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        kernel_size=9,
        dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                kernel_size=kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, phoneme_tokens, mask=None):
        """
        Args:
            phoneme_tokens: (B, S) containing phoneme token indices
            mask: optional attention mask
        Returns:
            (B, S, E) encoded phoneme representations
        """
        x = self.embedding(phoneme_tokens)  # (B, S, E)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class ConditionalLayerNorm(nn.Module):
    """
    LayerNorm that can incorporate an external conditioning signal (e.g., diffusion time embedding).
    """
    def __init__(self, normalized_shape, conditioning_dim, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # The base LayerNorm parameters (without affine)
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        # Learnable transforms (scale, shift) from the conditioning
        self.cond_scale = nn.Linear(conditioning_dim, normalized_shape[0])
        self.cond_shift = nn.Linear(conditioning_dim, normalized_shape[0])

    def forward(self, x, cond):
        """
        x: (B, S, D)
        cond: (B, conditioning_dim) - e.g., time embedding
        """
        # Standard layer norm (mean-only, then var-only normalization)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) / torch.sqrt(var + self.eps)

        # B, D
        scale = self.cond_scale(cond)
        shift = self.cond_shift(cond)

        # shape broadcast: (B, 1, D)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        return x_normed * (1 + scale) + shift


class ConditionalTransformerBlock(nn.Module):
    """
    A Transformer block that uses self-attention, convolutional feedforward,
    and conditional layer normalization.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, conditioning_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cond_ln1 = ConditionalLayerNorm(embed_dim, conditioning_dim)
        self.cond_ln2 = ConditionalLayerNorm(embed_dim, conditioning_dim)

        self.feedforward = ConvolutionalFeedForward(embed_dim, ff_dim, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.cond_ln1(x, cond)  # conditional layer norm

        # Feedforward with conv
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.cond_ln2(x, cond)  # conditional layer norm
        return x


class TimeEmbedding(nn.Module):
    """
    Simple sinusoidal or learnable embedding for diffusion time steps.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        # t is (B,) or (B, 1), a float representing the normalized diffusion time
        # Convert t to shape (B, embed_dim)
        half_dim = self.embed_dim // 2
        # Example: sinusoidal time embedding
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, embed_dim)
        
        # Optionally pass it through MLP
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb


class PhonemeProsodyDurationDiffusion(nn.Module):
    """
    Phoneme-level Transformer for predicting [avg_pitch, std_pitch, avg_energy, std_energy, duration]
    from phoneme encoder outputs.

    Architecture:
      - Input: (B, S, input_dim) from phoneme encoder (e.g., input_dim=512)
      - 1x Linear -> 1024
      - 6-layer Transformer (8 heads, 1024 embed, 2048 FF, kernel=3, dropout=0.1)
        with conditional LN for diffusion time
      - 1x Linear -> 5 features per phoneme
    """
    def __init__(
        self,
        input_dim=256,     # dimension of phoneme encoder output
        output_dim=2,
        embed_dim=1024,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Project phoneme encoder output (e.g. 512) -> 1024
        self.input_linear = nn.Linear(input_dim, embed_dim)

        # Time embedding for conditional LN
        self.time_embed = TimeEmbedding(embed_dim)

        # 6-layer Transformer
        self.layers = nn.ModuleList([
            ConditionalTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                conditioning_dim=embed_dim,
                kernel_size=3,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output: 2 features per phoneme
        # [duration, prosody]
        self.output_linear = nn.Linear(embed_dim, output_dim)


    def forward(self, phoneme_enc, t):
        """
        phoneme_enc: (B, S, input_dim) - output from phoneme encoder
        t: (B,) diffusion time steps (float or int)
        
        Returns:
          predicted_features: (B, S, 5)
        """
        # 1) Project to embed_dim=1024
        x = self.input_linear(phoneme_enc)  # (B, S, 1024)

        # 2) Get diffusion time embeddings
        time_emb = self.time_embed(t)       # (B, 1024)

        # 3) Pass through 6-layers of conditional Transformer
        for layer in self.layers:
            x = layer(x, time_emb)

        # 4) Linear -> (B, S, 5)
        predicted = self.output_linear(x)
        return predicted


class LengthRegulator(nn.Module):
    """
    A simple length regulator that expands each phoneme embedding according
    to its predicted duration, producing a frame-level representation.

    Example Usage:
      1) durations = duration_diffusion_model(...)  # (B, S) integer durations
      2) c_ph = length_regulator(phoneme_enc, durations)
         --> c_ph is (B, N_max, E), expanded and padded to the max length in the batch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, phoneme_enc, durations):
        """
        Args:
            phoneme_enc: (B, S, E) - phoneme-level encoder outputs
            durations: (B, S) - integer durations for each phoneme

        Returns:
            c_ph: (B, N_max, E) - frame-level embeddings, where each phoneme
                 is replicated according to its duration. Sequences are padded
                 to the max sequence length in the batch.
        """
        B, S, E = phoneme_enc.shape
        expanded_list = []

        for b in range(B):
            ph_enc_b = phoneme_enc[b]  # (S, E)
            dur_b = durations[b]       # (S,)

            frames_b = []
            for s in range(S):
                repeat_len = dur_b[s].item()
                if repeat_len > 0:
                    # replicate phoneme 's' repeat_len times
                    repeated = ph_enc_b[s].unsqueeze(0).repeat(int(repeat_len), 1)  # (repeat_len, E)
                    frames_b.append(repeated)
                # If repeat_len=0, we skip that phoneme (rare, but can happen).

            if len(frames_b) > 0:
                frames_b = torch.cat(frames_b, dim=0)  # (N_frames_b, E)
            else:
                # fallback if all durations == 0
                frames_b = torch.zeros((1, E), device=phoneme_enc.device)

            expanded_list.append(frames_b)

        # Now we pad each sample to the same length
        c_ph = nn.utils.rnn.pad_sequence(expanded_list, batch_first=True, padding_value=0.0)
        # c_ph: (B, N_max, E)

        return c_ph


def partial_mask(target_tokens, t, mask_id=9999):
    """
    Partially mask the target prosody tokens at ratio p = sigma(t).
    
    Args:
        target_tokens: (B, R, N) discrete tokens for the target prosody sequence
        t: (B,) discrete or continuous time steps, e.g. in [0, T]
        mask_id: int ID used for [MASK] token
        
    Returns:
        masked_tokens: (B, N) partially masked tokens
    """
    # Copy to avoid modifying in-place
    masked_tokens = target_tokens.clone()

    # For each sample in the batch, compute a fraction p = sigma(t_i)
    I, B, N = target_tokens.shape
    for i in range(I):
        for b in range(B):
            p_i = sin_schedule(t[b])

            rand_vals = torch.rand(N, device=target_tokens.device)
            mask_positions = rand_vals < p_i
            
            # Replace them with the mask_id
            masked_tokens[i, b, mask_positions] = mask_id

    return masked_tokens


def sin_schedule(step):
    """
    Example schedule: sigma(t) = sin(pi * t / (2 * T)).
    
    You can adapt this to your actual 't' range or an existing function.
    """
    # Assume 'step' is a float in [0, T], for demonstration
    T = 1000.0  # or pass it in some other way
    fraction = step / T
    p = math.sin(math.pi * fraction / 2)
    return p


class MultiStreamCrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, conditioning_dim, dropout=0.1):
        """
        Args:
            embed_dim: dimension of the main stream x
            num_heads: number of attention heads
            ff_dim: feed-forward dimension
            conditioning_dim: dimension for the conditional LN (often = embed_dim)
            dropout: dropout rate
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn3 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn4 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Conditional LN after each step (self-attn, cross1, cross2, cross3, cross4, feed-forward)
        self.cond_ln1 = ConditionalLayerNorm(embed_dim, conditioning_dim=conditioning_dim)
        self.cond_ln2 = ConditionalLayerNorm(embed_dim, conditioning_dim=conditioning_dim)
        self.cond_ln3 = ConditionalLayerNorm(embed_dim, conditioning_dim=conditioning_dim)
        self.cond_ln4 = ConditionalLayerNorm(embed_dim, conditioning_dim=conditioning_dim)
        self.cond_ln5 = ConditionalLayerNorm(embed_dim, conditioning_dim=conditioning_dim)
        self.cond_ln6 = ConditionalLayerNorm(embed_dim, conditioning_dim=conditioning_dim)

        # Convolutional feed-forward block
        self.ff = ConvolutionalFeedForward(embed_dim, ff_dim, kernel_size=3, dropout=dropout)

    def forward(
        self,
        x,              # (B, Nx, E) main tokens
        time_emb,       # (B, E) diffusion time or global condition
        cond1=None,     # (B, N1, E)
        cond2=None,     # (B, N2, E)
        cond3=None,     # (B, N3, E)
        cond4=None,     # (B, N4, E)
        x_mask=None,
        c1_mask=None,
        c2_mask=None,
        c3_mask=None,
        c4_mask=None
    ):
        """
        Executes:
          1. Self-attn + LN
          2. Cross-attn on cond1 + LN
          3. Cross-attn on cond2 + LN
          4. Cross-attn on cond3 + LN
          5. Cross-attn on cond4 + LN
          6. Feed-forward + LN
        """
        # 1) Self-attention on x
        sa_out, _ = self.self_attn(x, x, x, attn_mask=x_mask)
        x = x + sa_out
        x = self.cond_ln1(x, time_emb)

        # 2) Cross-attn on cond1
        if cond1 is not None:
            ca1_out, _ = self.cross_attn1(x, cond1, cond1, attn_mask=c1_mask)
            x = x + ca1_out
            x = self.cond_ln2(x, time_emb)

        # 3) Cross-attn on cond2
        if cond2 is not None:
            ca2_out, _ = self.cross_attn2(x, cond2, cond2, attn_mask=c2_mask)
            x = x + ca2_out
            x = self.cond_ln3(x, time_emb)

        # 4) Cross-attn on cond3
        if cond3 is not None:
            ca3_out, _ = self.cross_attn3(x, cond3, cond3, attn_mask=c3_mask)
            x = x + ca3_out
            x = self.cond_ln4(x, time_emb)

        # 5) Cross-attn on cond4
        if cond4 is not None:
            ca4_out, _ = self.cross_attn4(x, cond4, cond4, attn_mask=c4_mask)
            x = x + ca4_out
            x = self.cond_ln5(x, time_emb)

        # 6) Feed-forward
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.cond_ln6(x, time_emb)

        return x


class MainDiffusionModel(nn.Module):
    """
    12-layer Transformer for prosody diffusion
    - Cross-attends to c_ph (frame-level phoneme condition) or speaker embeddings
    - Partial noising on prosody tokens
    """
    def __init__(
        self,
        output_dim=1,
        prosody_vocab_size=1025,
        embed_dim=256,
        num_heads=8,
        ff_dim=2048,
        num_layers=12,
        dropout=0.1,
    ):
        super().__init__()
        self.main_embedding = nn.Embedding(prosody_vocab_size, embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)

        # 12 cross-attention blocks
        self.layers = nn.ModuleList([
            MultiStreamCrossAttnBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                conditioning_dim=embed_dim,  # time-embedding dimension
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(embed_dim, output_dim)


    def forward(self, x_tokens, cond_1, cond_2, cond_3, cond_4, t, x_mask=None, c1_mask=None, c2_mask=None, c3_mask=None, c4_mask=None):
        """
        x_tokens: main stream of prosody tokens [prompt + target]
        c_ph: (B, N_frames, E) frame-level phoneme embeddings
        cond_1: (B, 1, E) or (B, E) speaker style
        t: time step
        """
        # Possibly embed t in a separate LN or an additional cross condition
        x = self.main_embedding(x_tokens)
        time_emb = self.time_embed(t)  # (B, E)
        
        for layer in self.layers:
            x = layer(
                x,
                time_emb,
                cond1=cond_1,
                cond2=cond_2,
                cond3=cond_3,
                cond4=cond_4,
                x_mask=x_mask,
                c1_mask=c1_mask,
                c2_mask=c2_mask,
                c3_mask=c3_mask,
                c4_mask=c4_mask,
            )

        # final output
        logits = self.output_linear(x)
        return x, logits



class SpeechDiffusionModel(nn.Module):
    """
    12-layer Transformer for prosody diffusion
    - Cross-attends to c_ph (frame-level phoneme condition) or speaker embeddings
    - Partial noising on prosody tokens
    """
    def __init__(
        self,
        output_dim=1,
        prosody_vocab_size=1025,
        embed_dim=256,
        num_heads=8,
        ff_dim=2048,
        num_layers=12,
        dropout=0.1,
    ):
        super().__init__()
        self.main_embedding = nn.Linear(prosody_vocab_size, embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)

        # 12 cross-attention blocks
        self.layers = nn.ModuleList([
            MultiStreamCrossAttnBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                conditioning_dim=embed_dim,  # time-embedding dimension
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(embed_dim, output_dim)


    def forward(self, x_tokens, cond_1, cond_2, cond_3, cond_4, t, x_mask=None, c1_mask=None, c2_mask=None, c3_mask=None, c4_mask=None):
        """
        x_tokens: main stream of prosody tokens [prompt + target]
        c_ph: (B, N_frames, E) frame-level phoneme embeddings
        cond_1: (B, 1, E) or (B, E) speaker style
        t: time step
        """
        # Possibly embed t in a separate LN or an additional cross condition
        # x = self.main_embedding(x_tokens)
        time_emb = self.time_embed(t)  # (B, E)
        
        for layer in self.layers:
            x = layer(
                x_tokens,
                time_emb,
                cond1=cond_1,
                cond2=cond_2,
                cond3=cond_3,
                cond4=cond_4,
                x_mask=x_mask,
                c1_mask=c1_mask,
                c2_mask=c2_mask,
                c3_mask=c3_mask,
                c4_mask=c4_mask,
            )

        return x


def average_prosody(target_prosody, phoneme_cum_durations):
    num_phonemes = phoneme_cum_durations.shape[-1]
    _, batch_size, _ = target_prosody.shape
    averaged_prosody = torch.zeros(batch_size, num_phonemes)
    for b in range(batch_size):  # Iterate over batch
        prev_duration = 0
        for p in range(num_phonemes):  # Iterate over phonemes
                duration = phoneme_cum_durations[b][p]
                start_idx = int(prev_duration)
                end_idx = int(duration)
                if start_idx < end_idx:
                    averaged_prosody[b, p] = target_prosody[:, b, start_idx:end_idx].float().mean()
                prev_duration = duration
    return averaged_prosody


class NaturalSpeech3(nn.Module):
    def __init__(
        self,
        facodec_encoder: FACodecEncoder,
        facodec_decoder: FACodecDecoder,
        vocab_size=512,
        T=1000,
    ):
    
        super().__init__()
        self.T = T
        self.facodec_encoder = facodec_encoder
        self.facodec_decoder = facodec_decoder

        self.phoneme_encoder = PhonemeEncoder(vocab_size=vocab_size)
        self.duration_prosody_predictor = PhonemeProsodyDurationDiffusion()

        self.length_regulator = LengthRegulator()

        self.prosody_diffusion = MainDiffusionModel()
        self.content_diffusion = MainDiffusionModel()
        self.acoustic_detail_diffusion = MainDiffusionModel()
        self.speech_diffusion = SpeechDiffusionModel()


    def forward(
        self,
        phoneme_sequence: list[int],
        phoneme_durations: list[float],
        prompt_audio: torch.tensor,
        target_audio: torch.tensor,
    ):        
        prompt_enc_out = self.facodec_encoder(prompt_audio)
        prompt_vq_post_emb, prompt_vq_id, _, prompt_quantized, prompt_spk_embs = self.facodec_decoder(
            prompt_enc_out, eval_vq=True, vq=True
        )

        prompt_prosody = prompt_vq_id[:1]
        prompt_content = prompt_vq_id[1:3]
        prompt_acoustic_detail = prompt_vq_id[3:]

        target_enc_out = self.facodec_encoder(target_audio)

        target_vq_post_emb, target_vq_id, _, target_quantized, target_spk_embs = self.facodec_decoder(
            target_enc_out, eval_vq=True, vq=True
        )

        target_prosody = target_vq_id[:1]
        target_content = target_vq_id[1:3]
        target_acoustic_detail = target_vq_id[3:]

        phoneme_cum_durations = torch.cumsum(phoneme_durations, dim=1)
        prosody = average_prosody(target_prosody, phoneme_cum_durations)
        
        phoneme_enc = self.phoneme_encoder(phoneme_sequence)

        b, *_ = prompt_audio.shape
        t = torch.randint(low=0, high=self.T, size=(b,), dtype=torch.long)
        phoneme_prosody_duration = self.duration_prosody_predictor(phoneme_enc, t)

        durations_pred, prosody_pred = phoneme_prosody_duration[:,:,0], phoneme_prosody_duration[:,:,1:]

        c_ph = self.length_regulator(phoneme_enc, phoneme_durations)
        
        partial_mask_prosody = partial_mask(target_prosody, t, mask_id=1024)
        prosody_tokens = torch.cat([partial_mask_prosody, prompt_prosody],-1)
        prosody_tokens = prosody_tokens.permute(1, 0, 2)

        partial_mask_content = partial_mask(target_content, t, mask_id=1024)
        content_tokens = torch.cat([partial_mask_content, prompt_content],-1)
        content_tokens = content_tokens.permute(1, 0, 2)

        partial_mask_acoustic_detail = partial_mask(target_acoustic_detail, t, mask_id=1024)
        acoustic_detail_tokens = torch.cat([partial_mask_acoustic_detail, prompt_acoustic_detail],-1)
        acoustic_detail_tokens = acoustic_detail_tokens.permute(1, 0, 2)

        target_vq_post_emb = target_vq_post_emb.permute(1, 0, 2)
        prompt_vq_post_emb = prompt_vq_post_emb.permute(1, 0, 2)
        partial_mask_vq = partial_mask(target_vq_post_emb, t, mask_id=1024)
        vq_tokens = torch.cat([partial_mask_vq, prompt_vq_post_emb],-1)
        vq_tokens = vq_tokens.permute(1, 2, 0)

        original_prosody_second_dim = prosody_tokens.shape[-2]
        original_content_second_dim = content_tokens.shape[-2]
        original_acoustic_detail_second_dim = acoustic_detail_tokens.shape[-2]

        prosody_tokens = prosody_tokens.reshape(prosody_tokens.size(0), -1)
        content_tokens = content_tokens.reshape(content_tokens.size(0), -1)
        acoustic_detail_tokens = acoustic_detail_tokens.reshape(acoustic_detail_tokens.size(0), -1)

        zp, zp_l = self.prosody_diffusion(prosody_tokens, cond_1=c_ph, cond_2=None, cond_3=None, cond_4=None, t=t)
        zc, zc_l = self.content_diffusion(content_tokens, cond_1=c_ph, cond_2=zp, cond_3=None, cond_4=None,t=t)
        zd, zd_l = self.acoustic_detail_diffusion(acoustic_detail_tokens, cond_1=c_ph, cond_2=zp, cond_3=zc, cond_4=None, t=t)
        speech = self.speech_diffusion(vq_tokens, cond_1=c_ph, cond_2=zp, cond_3=zc, cond_4=zd, t=t)
        
        zp_l = zp_l.reshape(zc_l.size(0), original_prosody_second_dim, -1)
        zc_l = zc_l.reshape(zc_l.size(0), original_content_second_dim, -1)
        zd_l = zd_l.reshape(zc_l.size(0), original_acoustic_detail_second_dim, -1)

        L_prompt = prompt_prosody.shape[-1]

        zp_l = zp_l.permute(1, 0, 2)
        zc_l = zc_l.permute(1, 0, 2)
        zd_l = zd_l.permute(1, 0, 2)
        speech = speech.permute(2, 0, 1)
        
        padded_prosody_target = torch.full(
            zp_l.shape,
            fill_value=0,
            dtype=torch.long,
            device=zp_l.device
        )
        padded_prosody_target[:,:, L_prompt:] = target_prosody

        padded_content_target = torch.full(
            zc_l.shape,
            fill_value=0,
            dtype=torch.long,
            device=zc_l.device
        )
        padded_content_target[:,:, L_prompt:] = target_content

        padded_acoustic_detail_target = torch.full(
            zd_l.shape,
            fill_value=0,
            dtype=torch.long,
            device=zd_l.device
        )
        padded_acoustic_detail_target[:,:, L_prompt:] = target_acoustic_detail

        padded_vq_target = torch.full(
            speech.shape,
            fill_value=0,
            dtype=speech.dtype,
            device=speech.device
        )
        padded_vq_target[:,:, L_prompt:] = target_vq_post_emb
        
        duration_loss = F.l1_loss(phoneme_durations, durations_pred)
        prosody_loss = F.l1_loss(prosody, prosody_pred)
        zp_loss = F.l1_loss(padded_prosody_target, zp_l)
        zc_loss = F.l1_loss(padded_content_target, zc_l)
        zd_loss = F.l1_loss(padded_acoustic_detail_target, zd_l)
        speech_loss = F.l1_loss(padded_vq_target, speech)

        overall_loss = duration_loss + prosody_loss + zp_loss + zc_loss + zd_loss + speech_loss

        return overall_loss