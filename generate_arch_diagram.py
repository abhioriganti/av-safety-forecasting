import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── colour palette ──────────────────────────────────────────────────────────
C_INPUT    = "#1a6b4a"   # dark green
C_PROJ     = "#1a4a6b"   # dark blue
C_POSENC   = "#2d4a6b"   # medium blue
C_ENCODER  = "#4a2d6b"   # purple
C_ATTN     = "#4a3d1a"   # dark amber (attention sub-block)
C_FF       = "#1a3d4a"   # teal
C_POOL     = "#6b2d4a"   # dark rose
C_HEAD     = "#1a5a3a"   # green
C_CONF     = "#5a4a1a"   # gold
C_OUTPUT   = "#3a1a6b"   # deep purple
C_ATTN_OPT = "#4a1a1a"   # dark red (optional)
TEXT_LIGHT = "#e8e8e8"
TEXT_DIM   = "#aaaaaa"
ARROW_COL  = "#888888"
BORDER_COL = "#444444"


def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=11, radius=0.25):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=1.2, edgecolor=BORDER_COL,
        facecolor=color, zorder=3
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TEXT_LIGHT, zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=8, color=TEXT_DIM, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TEXT_LIGHT, zorder=4)


def arrow(ax, x, y_start, y_end, color=ARROW_COL):
    ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14), zorder=2)


def split_arrow(ax, x_src, y_src, targets, color=ARROW_COL):
    mid_y = y_src - 0.55
    ax.plot([x_src, x_src], [y_src, mid_y], color=color, lw=1.5, zorder=2)
    for (xt, yt) in targets:
        ax.plot([x_src, xt], [mid_y, mid_y], color=color, lw=1.5, zorder=2)
        ax.annotate("", xy=(xt, yt), xytext=(xt, mid_y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.5, mutation_scale=14), zorder=2)


# ── Title ───────────────────────────────────────────────────────────────────
ax.text(8, 21.4, "TrajectoryTransformer — Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=TEXT_LIGHT)
ax.text(8, 21.0, "AV Safety Forecasting  ·  Argoverse 2  ·  DATA 612",
        ha="center", va="center", fontsize=10, color=TEXT_DIM)

# ── (1) Input ────────────────────────────────────────────────────────────────
y = 20.0
box(ax, 8, y, 7, 0.65, C_INPUT,
    "Input Sequence",
    "[batch, T=50, 6]   ·   dx  dy  vx  vy  sin θ  cos θ")
arrow(ax, 8, y - 0.33, y - 0.82)

# ── (2) Linear Projection ────────────────────────────────────────────────────
y = 18.9
box(ax, 8, y, 7, 0.65, C_PROJ,
    "Linear Projection  (input_proj)",
    "nn.Linear(6 → 256)     →     [batch, 50, 256]")
arrow(ax, 8, y - 0.33, y - 0.82)

# ── (3) Positional Encoding ──────────────────────────────────────────────────
y = 17.8
box(ax, 8, y, 7, 0.65, C_POSENC,
    "Sinusoidal Positional Encoding",
    "sin / cos at 128 frequencies  ·  added in-place  →  [batch, 50, 256]")
arrow(ax, 8, y - 0.33, y - 0.7)

# ── (4) Transformer Encoder block (repeated × 4) ─────────────────────────────
enc_top = 17.0
enc_bot = 12.2
enc_h   = enc_top - enc_bot
enc_cy  = (enc_top + enc_bot) / 2

enc_bg = FancyBboxPatch(
    (1.5, enc_bot - 0.1), 13, enc_h + 0.2,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    linewidth=1.5, edgecolor="#6655aa", linestyle="--",
    facecolor="#1a1530", zorder=1
)
ax.add_patch(enc_bg)
ax.text(8, enc_top + 0.15, "Transformer Encoder   ×   4 layers",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="#9988dd", zorder=4)

# sub-blocks inside encoder
sub_y = [16.55, 15.55, 14.55, 13.55]
sub_labels = [
    ("Multi-Head Self-Attention", "4 heads  ·  d_model=256  ·  dropout=0.1"),
    ("Add & LayerNorm",           "residual  +  normalize"),
    ("Feed-Forward Network",      "Linear(256→512) → ReLU → Linear(512→256)  ·  dropout=0.1"),
    ("Add & LayerNorm",           "residual  +  normalize"),
]
sub_colors = [C_ATTN, C_POSENC, C_FF, C_POSENC]

for i, (yb, (lab, sub), col) in enumerate(zip(sub_y, sub_labels, sub_colors)):
    box(ax, 8, yb, 11.5, 0.65, col, lab, sub, fontsize=10)
    if i < len(sub_y) - 1:
        arrow(ax, 8, yb - 0.33, sub_y[i + 1] + 0.33)

# arrow into and out of encoder block
arrow(ax, 8, 17.8 - 0.33, sub_y[0] + 0.33)
arrow(ax, 8, sub_y[-1] - 0.33, enc_bot - 0.45)

ax.text(8, enc_bot - 0.28, "Output: [batch, 50, 256]",
        ha="center", va="center", fontsize=9, color=TEXT_DIM, zorder=4)

# ── (5) Mean Pooling ─────────────────────────────────────────────────────────
y = 11.5
arrow(ax, 8, enc_bot - 0.45, y + 0.33)
box(ax, 8, y, 7, 0.65, C_POOL,
    "Mean Pool  (across T=50 timesteps)",
    "[batch, 50, 256]   →   [batch, 256]   ·   preserves full temporal context")

# ── split arrow to two heads ─────────────────────────────────────────────────
split_arrow(ax, 8, y - 0.33,
            [(4.5, 10.02), (11.5, 10.02)],
            color=ARROW_COL)

# ── (6a) Mode Head ───────────────────────────────────────────────────────────
y_h = 9.6
box(ax, 4.5, y_h + 0.4, 5.5, 0.65, C_HEAD,
    "Mode Head",
    "Linear(256→256) → ReLU → Linear(256→720)")
arrow(ax, 4.5, y_h + 0.07, y_h - 0.52)
box(ax, 4.5, y_h - 0.85, 5.5, 0.55, C_HEAD,
    "Reshape",
    "[batch, 6×60×2]  →  [batch, 6, 60, 2]", fontsize=9)
arrow(ax, 4.5, y_h - 1.12, y_h - 1.65)
box(ax, 4.5, y_h - 1.95, 5.5, 0.55, C_OUTPUT,
    "Trajectory Predictions",
    "[batch, K=6, T=60, 2 (x,y)]", fontsize=9)

# ── (6b) Confidence Head ─────────────────────────────────────────────────────
box(ax, 11.5, y_h + 0.4, 5.5, 0.65, C_CONF,
    "Confidence Head",
    "Linear(256 → 6)   →   [batch, 6]  (raw logits)")
arrow(ax, 11.5, y_h + 0.07, y_h - 0.67)
box(ax, 11.5, y_h - 0.95, 5.5, 0.55, C_OUTPUT,
    "Mode Confidences",
    "[batch, K=6]  ·  softmax at inference", fontsize=9)

# ── (7) Optional: Attention Extraction ───────────────────────────────────────
y_attn = 6.5
ax.annotate("", xy=(8, y_attn + 0.33), xytext=(8, 11.5 - 0.33),
            arrowprops=dict(arrowstyle="-|>", color="#dd4444",
                            lw=1.2, linestyle="dashed", mutation_scale=12), zorder=2)
box(ax, 8, y_attn, 8.5, 0.65, C_ATTN_OPT,
    "Attention Extraction  (optional, interpretability only)",
    "Separate MHA layer  ·  [batch, 50, 50]  ·  which timesteps drove prediction")
ax.text(9.5, y_attn + 0.5, "return_attention=True", fontsize=8,
        color="#dd8888", ha="left", va="center", style="italic")

# ── (8) WTA Training note ────────────────────────────────────────────────────
y_wta = 5.3
note_bg = FancyBboxPatch(
    (1.2, y_wta - 0.65), 13.6, 1.1,
    boxstyle="round,pad=0.08,rounding_size=0.2",
    linewidth=1, edgecolor="#445544", facecolor="#141e14", zorder=1
)
ax.add_patch(note_bg)
ax.text(8, y_wta, "Training: Winner-Takes-All (WTA) Loss",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#88cc88")
ax.text(8, y_wta - 0.38,
        "Select mode k* = argmin FDE(pred_k, GT)  ·  apply Huber loss only to k*  ·  enforces mode diversity",
        ha="center", va="center", fontsize=8.5, color=TEXT_DIM)

# ── (9) Hyperparameter table ─────────────────────────────────────────────────
y_tbl = 4.1
tbl_bg = FancyBboxPatch(
    (0.5, y_tbl - 1.3), 15, 1.65,
    boxstyle="round,pad=0.08,rounding_size=0.2",
    linewidth=1, edgecolor="#334455", facecolor="#111a22", zorder=1
)
ax.add_patch(tbl_bg)
ax.text(8, y_tbl + 0.2, "Hyperparameters",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#88aacc")

params = [
    ("d_model", "256"),
    ("Heads", "4"),
    ("Layers", "4"),
    ("FFN dim", "512"),
    ("Dropout", "0.1"),
    ("Modes K", "6"),
    ("Obs T", "50"),
    ("Pred T", "60"),
    ("Batch", "256"),
    ("LR", "5e-4"),
]
col_w = 15.0 / len(params)
for i, (k, v) in enumerate(params):
    cx = 0.5 + col_w * (i + 0.5)
    ax.text(cx, y_tbl - 0.28, k, ha="center", va="center",
            fontsize=8, color=TEXT_DIM)
    ax.text(cx, y_tbl - 0.72, v, ha="center", va="center",
            fontsize=9, fontweight="bold", color=TEXT_LIGHT)

# ── (10) Metrics bar ─────────────────────────────────────────────────────────
y_met = 2.3
met_bg = FancyBboxPatch(
    (0.5, y_met - 0.55), 15, 0.9,
    boxstyle="round,pad=0.08,rounding_size=0.2",
    linewidth=1, edgecolor="#445533", facecolor="#141e11", zorder=1
)
ax.add_patch(met_bg)
ax.text(8, y_met + 0.15, "Test Set Metrics (Argoverse 2)",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#88cc66")
ax.text(8, y_met - 0.22,
        "minADE@6  =  1.44 m          minFDE@6  =  3.28 m          Trained: 75 epochs  ·  RTX 4060 8 GB  ·  AMP fp16",
        ha="center", va="center", fontsize=9, color=TEXT_DIM)

# ── (11) Legend ──────────────────────────────────────────────────────────────
legend_items = [
    (C_INPUT,    "Input / Features"),
    (C_PROJ,     "Linear Projection"),
    (C_POSENC,   "Pos. Encoding / Norm"),
    (C_ATTN,     "Self-Attention"),
    (C_FF,       "Feed-Forward"),
    (C_POOL,     "Pooling"),
    (C_HEAD,     "Mode Head"),
    (C_CONF,     "Confidence Head"),
    (C_OUTPUT,   "Output"),
    (C_ATTN_OPT, "Optional (interpretability)"),
]
patches = [mpatches.Patch(facecolor=c, edgecolor=BORDER_COL, label=l)
           for c, l in legend_items]
ax.legend(handles=patches, loc="lower center", ncol=5,
          framealpha=0.15, facecolor="#111111", edgecolor="#444444",
          fontsize=8, labelcolor=TEXT_LIGHT,
          bbox_to_anchor=(0.5, 0.0))

plt.tight_layout()
out = r"C:\Users\abhis\projects\av_safety\outputs\architecture_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
