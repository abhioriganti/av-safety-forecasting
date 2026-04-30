import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TrajectoryTransformer(nn.Module):
    """
    Transformer encoder for AV trajectory forecasting.

    Changes from v1:
    - Mean-pools all encoder hidden states instead of using only the last token.
      This preserves more temporal context across the 50-step history.
    - Outputs K=6 possible future trajectories (multimodal prediction).
      minADE / minFDE over K modes is the standard Argoverse 2 metric.
    - Returns attention weights from the final encoder layer so callers
      can visualise which past timesteps drove the prediction (interpretability).
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pred_len: int = 60,
        num_modes: int = 6,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.num_modes = num_modes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Standard encoder layers (no need_weights here; we add a separate
        # attention layer below just for interpretability extraction)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Separate single attention layer used only for weight extraction
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True,
        )

        # Prediction head: outputs K modes, each with pred_len (x,y) pairs
        # plus a confidence score per mode
        self.mode_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modes * pred_len * 2),
        )
        self.confidence_head = nn.Linear(d_model, num_modes)

    def forward(self, x, return_attention: bool = False):
        """
        Args:
            x: [batch, obs_len, input_dim]
            return_attention: if True, also return attention weights

        Returns:
            preds:       [batch, num_modes, pred_len, 2]
            confidences: [batch, num_modes]  (raw logits; use softmax outside)
            attn_weights (optional): [batch, obs_len, obs_len]
        """
        x_proj = self.input_proj(x)           # [B, T, d_model]
        x_enc = self.pos_encoder(x_proj)       # [B, T, d_model]
        h = self.encoder(x_enc)                # [B, T, d_model]

        # Mean-pool across time — keeps all temporal context
        h_pooled = h.mean(dim=1)               # [B, d_model]

        # Multimodal predictions
        raw = self.mode_head(h_pooled)         # [B, K * pred_len * 2]
        preds = raw.view(x.size(0), self.num_modes, self.pred_len, 2)
        confidences = self.confidence_head(h_pooled)  # [B, K]

        if return_attention:
            # Run a single attention pass on the encoder output for weight extraction
            _, attn_weights = self.attn_layer(h, h, h, need_weights=True)
            return preds, confidences, attn_weights

        return preds, confidences
