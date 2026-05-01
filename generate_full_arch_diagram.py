import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(22, 34))
W, H = 22, 34
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis("off")
BG = "#0c0e18"
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# ── Palette ──────────────────────────────────────────────────────────────────
C_DATA  = "#14503c"
C_TRANS = "#182060"
C_FEAT  = "#503c14"
C_CLF   = "#501428"
C_LBL   = "#142850"
C_RAG   = "#2c1450"
C_LLM   = "#3c2808"
C_OUT   = "#143c14"
C_TRAIN = "#0c160c"
C_GT    = "#18182c"
TXT     = "#e0e0e0"
DIM     = "#888888"
ARR     = "#7788aa"
BRD     = "#404050"


def B(cx, cy, w, h, col, t1, t2=None, fs1=10, fs2=8, rad=0.22,
      ec=None, ls="-", tc=None, dc=None):
    ec = ec or BRD; tc = tc or TXT; dc = dc or DIM
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle=f"round,pad=0.06,rounding_size={rad}",
        lw=1.3, edgecolor=ec, facecolor=col, linestyle=ls, zorder=3))
    dy = h * 0.18 if t2 else 0
    ax.text(cx, cy+dy, t1, ha="center", va="center", fontsize=fs1,
            fontweight="bold", color=tc, zorder=5)
    if t2:
        ax.text(cx, cy - h*0.22, t2, ha="center", va="center",
                fontsize=fs2, color=dc, zorder=5)


def VA(x, y0, y1, col=ARR, lw=1.5):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=14), zorder=2)


def L(x0, y0, x1, y1, col=ARR, lw=1.5, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=col, lw=lw, linestyle=ls, zorder=2)


def SH(y, label, col):
    ax.text(0.5, y, label, fontsize=8.5, fontweight="bold",
            color=col, va="center", zorder=5)
    L(3.5, y, W-0.5, y, col=col, lw=0.4, ls="--")


# ── TITLE ────────────────────────────────────────────────────────────────────
ax.text(W/2, 33.5, "AV Safety Forecasting — Complete System Architecture",
        ha="center", va="center", fontsize=17, fontweight="bold", color=TXT)
ax.text(W/2, 33.0,
        "Argoverse 2  ·  TrajectoryTransformer  +  RandomForest Safety Classifier"
        "  +  Llama-3.2 LLM Diagnosis  ·  DATA 612",
        ha="center", va="center", fontsize=10, color=DIM)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA
# ═════════════════════════════════════════════════════════════════════════════
SH(32.45, "MODULE 1 — DATA INPUT", "#66bbff")

B(11, 31.95, 20, 0.7, C_DATA,
  "Argoverse 2 Motion Forecasting Dataset",
  "224,896 driving scenarios   10 Hz sampling   urban environments   train / val / test splits")

# Split to X_obs and Y_gt
L(11, 31.6, 11, 31.3); L(11, 31.3, 6, 31.3); L(11, 31.3, 16, 31.3)
VA(6, 31.3, 31.0); VA(16, 31.3, 31.0)

B(6, 30.65, 9.5, 0.6, C_TRANS,
  "X_obs  (past context)",
  "[N, T=50, 6]    dx  dy  vx  vy  sin_theta  cos_theta", fs2=8)

B(16, 30.65, 8.5, 0.6, C_GT,
  "Y_gt  (ground truth futures - training only)",
  "[N, T=60, 2]    future x/y positions",
  ec="#556677", ls="--", tc="#8899aa", dc="#667788", fs2=8)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — TRAJECTORY FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
SH(30.1, "MODULE 2 — TRAJECTORY FORECASTING", "#aa88ff")

VA(6, 30.35, 29.65)

# TrajectoryTransformer collapsed block
ax.add_patch(FancyBboxPatch(
    (1.5, 27.5), 9.5, 2.05,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    lw=1.5, edgecolor="#7766cc", linestyle="--", facecolor="#10102a", zorder=1))
ax.text(6.25, 29.35, "TrajectoryTransformer", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="#bb99ff", zorder=4)
ax.text(6.25, 29.0,
        "Linear(6->256)  +  Sinusoidal Positional Encoding  [batch, 50, 256]",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(6.25, 28.65,
        "4 x TransformerEncoderLayer   (d_model=256, heads=4, ffn=512, dropout=0.1)",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(6.25, 28.3,
        "Mean Pool  ->  ModeHead [K=6, T=60, 2]  +  ConfidenceHead [K=6]",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(6.25, 27.65, "see detail diagram: outputs/architecture_diagram.png",
        ha="center", va="center", fontsize=8, color="#7766cc", style="italic", zorder=4)

# WTA Loss dashed arrow from Transformer to Y_gt
ax.annotate("", xy=(16.0, 30.35), xytext=(11.0, 28.7),
            arrowprops=dict(arrowstyle="<-", color="#556677", lw=1.2,
                            linestyle="dashed", mutation_scale=10), zorder=2)
ax.text(14.2, 29.7, "WTA Huber Loss\n(training only)",
        ha="center", va="center", fontsize=7.5, color="#667788",
        style="italic", zorder=4)

# Attention interpretability optional output (dashed, right of transformer)
ax.annotate("", xy=(14.5, 28.3), xytext=(11.0, 28.3),
            arrowprops=dict(arrowstyle="-|>", color="#884444", lw=1.2,
                            linestyle="dashed", mutation_scale=10), zorder=2)
ax.add_patch(FancyBboxPatch((14.5, 27.85), 7, 0.85,
    boxstyle="round,pad=0.06,rounding_size=0.2",
    lw=1.2, edgecolor="#664444", linestyle="--", facecolor="#1a0a0a", zorder=3))
ax.text(18.0, 28.43, "Attention Interpretability  (optional)",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#cc8888", zorder=5)
ax.text(18.0, 28.1, "return_attention=True  ->  [batch, T=50, T=50]",
        ha="center", va="center", fontsize=7.5, color=DIM, zorder=5)
ax.text(18.0, 27.95, "top-3 attended past timesteps printed per sample",
        ha="center", va="center", fontsize=7.5, color=DIM, zorder=5)

VA(6.25, 27.5, 26.9)

B(6.25, 26.6, 10, 0.6, "#1a1a40",
  "Predicted Trajectories   [N, K=6, T=60, 2]",
  "6 candidate future modes  x  60 timesteps  x  (x, y) relative positions", fs2=8)

VA(6.25, 26.3, 25.7)

B(6.25, 25.4, 10, 0.6, "#1a1a40",
  "Best Mode Selection",
  "k* = argmax( ConfidenceHead softmax )     ->     [N, T=60, 2]", fs2=8)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — KINEMATIC FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
SH(24.95, "MODULE 3 — KINEMATIC FEATURE EXTRACTION", "#ffaa44")

VA(6.25, 25.1, 24.55)

B(11, 24.15, 20, 0.7, C_FEAT,
  "Feature Extraction   (18 kinematic features per trajectory)",
  "max_speed   max_heading_change   heading_variance   max_lateral_dev   oscillation_score"
  "   max_accel   max_decel   max_jerk   path_efficiency   mean_curvature   ...", fs2=8)

# Weak supervision labeling (training only, right side)
L(11, 23.8, 11, 23.55); L(11, 23.55, 17.5, 23.55)
VA(17.5, 23.55, 23.25)
B(17.5, 22.95, 7.5, 0.6, C_GT,
  "_derive_label()  (training only)",
  "rule-based thresholds  ->  y in {0,1,2,3,4}   weak supervision",
  ec="#556677", ls="--", tc="#8899aa", dc="#667788", fs2=7.5)
VA(17.5, 22.65, 22.15)
B(17.5, 21.85, 7.5, 0.6, C_GT,
  "RF Training Labels   (training only)",
  "199,908 GT trajectories   balanced_subsample",
  ec="#556677", ls="--", tc="#8899aa", dc="#667788", fs2=7.5)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SAFETY CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
SH(23.55, "MODULE 4 — SAFETY CLASSIFICATION", "#ff6688")

VA(11, 23.8, 23.15)

B(11, 22.75, 16, 0.7, C_CLF,
  "SafetyClassifier  =  StandardScaler  +  RandomForestClassifier",
  "300 trees   balanced_subsample class weights   trained on 199,908 GT trajectories (weak supervision)", fs2=8)

VA(11, 22.4, 21.8)

B(11, 21.5, 16, 0.65, C_LBL,
  "Safety Label  +  Class Probabilities   [5]",
  "0=Safe (73.2%)    1=Sharp Turn (6.1%)    2=Oscillatory (17.7%)"
  "    3=High-Speed (2.9%)    4=Near-Collision (0.12%)", fs2=8)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — AUGMENTED DIAGNOSIS
# ═════════════════════════════════════════════════════════════════════════════
SH(21.0, "MODULE 5 — LLM DIAGNOSIS", "#cc88ff")

VA(11, 21.17, 20.5)

B(11, 20.1, 16, 0.7, C_LLM,
  "LLM Diagnosis — Llama-3.2-3B-Instruct",
  "4-bit NF4 quantization (bitsandbytes NF4)   ~1.7 GB VRAM   greedy decoding (do_sample=False)", fs2=8)

VA(11, 19.75, 19.15)

B(11, 18.85, 16, 0.65, C_LLM,
  "Structured JSON Prompt  ->  Parsed JSON Output",
  "Input: event_name + feature values + class_probs [5]     Output: event_type + severity + primary_indicator + secondary_indicators + recommended_action + confidence", fs2=8)

VA(11, 18.52, 17.85)

# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
SH(17.7, "OUTPUT — SAFETY DIAGNOSIS REPORT", "#66dd66")

B(11, 17.3, 18, 0.75, C_OUT,
  "Safety Diagnosis Report",
  "event_label   event_name   class_probs [5]   severity   primary_indicator"
  "   secondary_indicators   recommended_action   confidence", fs2=8.5)

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch((0.6, 14.6), W-1.2, 2.3,
    boxstyle="round,pad=0.08,rounding_size=0.25",
    lw=1, edgecolor="#334433", facecolor=C_TRAIN, zorder=1))
ax.text(11, 16.7, "Training Pipeline  (offline, one-time)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#88cc88", zorder=4)
ax.text(11, 16.3,
        "TrajectoryTransformer:  75 epochs  |  WTA Huber Loss on K=6 modes  |  "
        "early stopping (patience=15)  |  AMP fp16  |  AdamW lr=5e-4  |  RTX 4060 8 GB",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(11, 15.9,
        "SafetyClassifier:  _derive_label() rule-based weak supervision  ->  "
        "RandomForest(300 trees, balanced_subsample)  |  trained on 199,908 GT trajectories",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(11, 15.5,
        "Class imbalance 592:1  (Safe 73.2% vs Near-Collision 0.12%)  "
        "|  balanced_subsample compensates per tree  |  OOB acc ~1.0 on GT features",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(11, 15.1,
        "Distribution shift at inference:  model-predicted features drift ~400-900% from GT  "
        "(heading change +721%, jerk +885%, speed only +13%)",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)

# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch((0.6, 12.5), W-1.2, 1.8,
    boxstyle="round,pad=0.08,rounding_size=0.25",
    lw=1, edgecolor="#334422", facecolor="#0a1208", zorder=1))
ax.text(11, 14.1, "Evaluation Metrics",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#aadd88", zorder=4)
ax.text(11, 13.7,
        "Trajectory Forecasting:  minADE@6 = 1.44 m   |   minFDE@6 = 3.28 m"
        "   (Argoverse 2 test set, 4,998 scenarios)",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(11, 13.3,
        "Safety Classification (Q85, test set, model-predicted features vs GT labels):",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)
ax.text(11, 12.9,
        "High-Speed AUC-ROC=0.935  AUC-PR=0.759  F1=0.70   |   "
        "Oscillatory AUC-ROC=0.813  AUC-PR=0.358  F1=0.43   |   "
        "Sharp Turn AUC-PR=0.039 (heading drift kills detection)",
        ha="center", va="center", fontsize=8.5, color=DIM, zorder=4)

# ═════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_DATA,  "Data / Input"),
    (C_TRANS, "Trajectory Forecasting"),
    (C_FEAT,  "Feature Extraction"),
    (C_CLF,   "Safety Classifier"),
    (C_LBL,   "Output Labels"),
    (C_LLM,   "LLM Diagnosis"),
    (C_OUT,   "Final Report"),
    (C_GT,    "Training-only (dashed)"),
]
patches = [mpatches.Patch(facecolor=c, edgecolor=BRD, label=l)
           for c, l in legend_items]
ax.legend(handles=patches, loc="lower center", ncol=5,
          framealpha=0.15, facecolor="#0c0c14", edgecolor=BRD,
          fontsize=9, labelcolor=TXT, bbox_to_anchor=(0.5, 0.005))

plt.tight_layout()
out = r"C:\Users\abhis\projects\av_safety\outputs\full_architecture_diagram.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
