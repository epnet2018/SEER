"""
The paper's multi-task seismic-monitoring model (architecture in Fig. 3).
====================================================================
- shared residual self-attention encoder (4 input channels: 3 normalized
  waveforms + 1 log-amplitude channel), per-station representation h;
- station-level attention aggregation: per-station scores conditioned on
  epicentral distance and spectral features, softmax-pooled over the event's
  stations into an event-level representation z;
- event-level task heads: event type (natural vs non-natural) and magnitude;
- auxiliary station-level type head: structurally identical to the event type
  head (same input at K = 1); trained with loss weight W_ST to give the
  per-station representation h dense, event-label supervision;
- station-level first-motion polarity head (independent CNN on the first 2 s
  of the 3 channels);
- MC Dropout (``enable_mc_dropout``) provides the inference-time confidence
  estimator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import N_CH, N_CH_ENC, N_RAW_FEAT, N_DIST_FEAT, N_SPEC_FEAT
from building_blocks import ImprovedEncoder, PolarityDecoder, MagnitudeDecoder, TaskDecoder


class SeismicModel(nn.Module):
    """The paper's model: shared encoder + station-level attention + event-level heads
    + auxiliary station-level type head + station-level polarity head."""
    def __init__(self, attn_dim=512):
        super().__init__()
        self.encoder = ImprovedEncoder(in_ch=N_CH_ENC, base_ch=64)
        feat_dim = self.encoder.feat_dim                     # 1024
        meta_dim = N_DIST_FEAT + N_SPEC_FEAT                # 6
        self.attn_dim = attn_dim
        # station-level attention aggregation
        self.attn_V = nn.Linear(feat_dim, attn_dim)
        self.attn_M = nn.Linear(meta_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        # event-level heads: type + magnitude
        type_in = attn_dim + N_RAW_FEAT + N_DIST_FEAT + N_SPEC_FEAT
        self.type_decoder = TaskDecoder(type_in, [512, 256], 2)
        self.mag_decoder = MagnitudeDecoder(attn_dim)
        # auxiliary station-level type head: input identical to the event head at K=1
        self.station_type_decoder = TaskDecoder(type_in, [512, 256], 2)
        # station-level head: polarity
        self.polarity_decoder = PolarityDecoder(in_ch=N_CH, embed_dim=128)

    def forward(self, x, raw_feat, dist_feat, spec_feat, ev_idx, pos_idx, n_events, k_max,
                return_polarity=True):
        """
        Returns: pol_out [N,2], type_out [B,2], mag_out [B],
              st_type_out [N,2] (per-station), alpha [B,k_max]
        """
        N = x.size(0)
        enc = self.encoder(x)                                # [N, 1024]
        meta = torch.cat([dist_feat, spec_feat], dim=1)       # [N, 6]
        h = torch.tanh(self.attn_V(enc) + self.attn_M(meta))  # [N, A]
        s = self.attn_w(h).squeeze(-1)                        # [N]
        s_pad = s.new_full((n_events, k_max), -1e9)
        s_pad[ev_idx, pos_idx] = s
        alpha = F.softmax(s_pad, dim=1)                       # [B, K]
        h_pad = h.new_zeros(n_events, k_max, self.attn_dim)
        h_pad[ev_idx, pos_idx] = h
        z_ev = (alpha.unsqueeze(-1) * h_pad).sum(dim=1)       # [B, A]
        raw_pad = raw_feat.new_zeros(n_events, k_max, raw_feat.size(1))
        raw_pad[ev_idx, pos_idx] = raw_feat
        dist_pad = dist_feat.new_zeros(n_events, k_max, dist_feat.size(1))
        dist_pad[ev_idx, pos_idx] = dist_feat
        spec_pad = spec_feat.new_zeros(n_events, k_max, spec_feat.size(1))
        spec_pad[ev_idx, pos_idx] = spec_feat
        raw_ev = (alpha.unsqueeze(-1) * raw_pad).sum(dim=1)
        dist_ev = (alpha.unsqueeze(-1) * dist_pad).sum(dim=1)
        spec_ev = (alpha.unsqueeze(-1) * spec_pad).sum(dim=1)
        # event-level outputs
        type_out = self.type_decoder(torch.cat([z_ev, raw_ev, dist_ev, spec_ev], dim=1))
        mag_out = self.mag_decoder(z_ev, raw_ev, dist_ev)
        # auxiliary station-level type output
        st_type_out = self.station_type_decoder(
            torch.cat([h, raw_feat, dist_feat, spec_feat], dim=1))   # [N, 2]
        # station-level polarity
        pol_out = self.polarity_decoder(x[:, :N_CH, :200]) if return_polarity else None
        return pol_out, type_out, mag_out, st_type_out, alpha


def enable_mc_dropout(model):
    """Enable Dropout at inference time (MC Dropout uncertainty estimation)"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model
