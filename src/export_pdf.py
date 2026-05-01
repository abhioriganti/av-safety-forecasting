"""
export_pdf.py -- Export training history + demo results as a styled PDF.

Usage:
    cd C:\\Users\\abhis\\projects\\av_safety
    python src/export_pdf.py
Output: outputs/demo/results_report.pdf
"""

import json, pathlib, textwrap, re
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

ROOT    = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "outputs" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load saved data ───────────────────────────────────────────────────────────
history = json.loads((ROOT / "outputs" / "train_history.json").read_text())
results = json.loads((ROOT / "outputs" / "results.json").read_text())
demo    = json.loads((ROOT / "outputs" / "demo_reports.json").read_text())

macro_line = [l for l in results["test_safety_report"].splitlines() if "macro avg" in l][0]
nums = re.findall(r"\d+\.\d+", macro_line)
prec, rec, f1 = float(nums[0]), float(nums[1]), float(nums[2])

# ── knowledge base retrieval (same as show_results.py) ───────────────────────
FEATURE_KEYS = ["max_speed","mean_speed","max_heading_change","mean_heading_change",
                "heading_variance","max_lateral_dev","mean_lateral_dev",
                "total_distance","final_displacement","oscillation_score"]
KB = [
    {"features":[0.4,0.3,0.2,0.1,0.02,0.3,0.1,20,8,0.05],
     "text":"Safe trajectory with smooth steering and low lateral deviation throughout."},
    {"features":[0.6,0.5,0.3,0.15,0.03,0.5,0.2,30,12,0.08],
     "text":"Normal urban driving at moderate speed with minimal heading variance, safe lateral margins."},
    {"features":[0.8,0.6,1.5,0.8,0.4,1.2,0.6,25,10,0.15],
     "text":"Sharp right turn detected -- heading change exceeds 1.2 rad, consistent with aggressive cornering."},
    {"features":[0.7,0.55,1.8,1.0,0.55,1.5,0.7,22,9,0.18],
     "text":"Abrupt lane-change in a short horizon suggests a sharp turn or evasive maneuver."},
    {"features":[0.9,0.7,2.0,1.1,0.6,1.8,0.9,28,11,0.2],
     "text":"Large heading change in a short horizon suggests a sharp turn or evasive maneuver."},
    {"features":[0.6,0.5,0.9,0.5,0.6,0.8,0.4,28,6,0.45],
     "text":"Predicted trajectory oscillation can indicate uncertain or unsafe motion planning."},
    {"features":[0.5,0.4,0.7,0.4,0.55,0.7,0.35,22,5,0.5],
     "text":"Lateral sway pattern with high heading variance suggests oscillatory motion behavior."},
    {"features":[2.2,1.8,0.5,0.25,0.08,0.8,0.4,60,22,0.12],
     "text":"High-speed trajectory detected -- velocity exceeds safe urban driving threshold."},
    {"features":[2.5,2.0,0.6,0.3,0.1,1.0,0.5,70,26,0.14],
     "text":"Sustained high-speed motion with moderate heading change poses collision risk."},
    {"features":[2.0,1.6,0.8,0.4,0.15,2.8,1.4,55,20,0.2],
     "text":"Near-collision risk: high speed combined with large lateral deviation from expected path."},
]
KB_F = np.array([e["features"] for e in KB], dtype=np.float32)
KB_N = KB_F / (np.linalg.norm(KB_F, axis=1, keepdims=True) + 1e-8)

def retrieve(features, top_k=3):
    vec = np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    sims = (KB_N @ vec).tolist()
    ranked = sorted(zip(sims, KB), key=lambda x: -x[0])
    return [(round(s, 4), e["text"]) for s, e in ranked[:top_k]]

# ── colour palette ────────────────────────────────────────────────────────────
BG_DARK   = colors.HexColor("#1e1e2e")
BG_PANEL  = colors.HexColor("#2a2a3e")
BG_HEADER = colors.HexColor("#313155")
GREEN     = colors.HexColor("#50fa7b")
CYAN      = colors.HexColor("#8be9fd")
YELLOW    = colors.HexColor("#f1fa8c")
ORANGE    = colors.HexColor("#ffb86c")
PINK      = colors.HexColor("#ff79c6")
WHITE     = colors.HexColor("#f8f8f2")
GRAY      = colors.HexColor("#6272a4")
RED       = colors.HexColor("#ff5555")

EVENT_COLORS = {
    "Safe":               colors.HexColor("#50fa7b"),
    "Sharp Turn":         colors.HexColor("#ffb86c"),
    "Oscillatory Motion": colors.HexColor("#8be9fd"),
    "High-Speed Risk":    colors.HexColor("#ff5555"),
    "Near-Collision Risk":colors.HexColor("#ff79c6"),
}

# ── styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def mono(text, color=WHITE, size=8):
    return ParagraphStyle("mono", fontName="Courier", fontSize=size,
                          textColor=color, leading=size * 1.4,
                          backColor=BG_DARK, spaceAfter=0)

S_TITLE   = ParagraphStyle("title",  fontName="Helvetica-Bold", fontSize=18,
                            textColor=WHITE, alignment=TA_CENTER, spaceAfter=4)
S_SUB     = ParagraphStyle("sub",    fontName="Helvetica",      fontSize=10,
                            textColor=GRAY,  alignment=TA_CENTER, spaceAfter=16)
S_H1      = ParagraphStyle("h1",     fontName="Helvetica-Bold", fontSize=13,
                            textColor=CYAN,  spaceAfter=6, spaceBefore=14)
S_H2      = ParagraphStyle("h2",     fontName="Helvetica-Bold", fontSize=10,
                            textColor=YELLOW, spaceAfter=4, spaceBefore=8)
S_MONO    = ParagraphStyle("mono",   fontName="Courier",        fontSize=8,
                            textColor=WHITE, leading=11.5,
                            backColor=BG_DARK, leftIndent=6)
S_BODY    = ParagraphStyle("body",   fontName="Helvetica",      fontSize=9,
                            textColor=WHITE, leading=13, spaceAfter=4)
S_LABEL   = ParagraphStyle("label",  fontName="Helvetica-Bold", fontSize=9,
                            textColor=ORANGE, spaceAfter=3)
S_DIAG    = ParagraphStyle("diag",   fontName="Helvetica",      fontSize=9,
                            textColor=WHITE, leading=13, leftIndent=8,
                            backColor=BG_PANEL, spaceAfter=4)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=8, spaceBefore=4)


def section_header(text):
    return Paragraph(text, S_H1)


def code_line(text):
    return Paragraph(text.replace(" ", "&nbsp;"), S_MONO)

# ── document ──────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "results_report.pdf"
doc = SimpleDocTemplate(
    str(pdf_path), pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

story = []

# ── cover header ──────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("AV Safety Forecasting", S_TITLE))
story.append(Paragraph("TrajectoryTransformer + SafetyClassifier + LLM Diagnosis  |  Argoverse 2  |  DATA 612", S_SUB))
story.append(hr())

# ── model summary table ───────────────────────────────────────────────────────
story.append(section_header("Model Configuration"))
cfg_data = [
    ["Parameter",        "Value"],
    ["Architecture",     "TransformerEncoder (4 layers, 4 heads, d_model=256)"],
    ["Dataset",          "Argoverse 2 Motion Forecasting (199,908 train scenarios)"],
    ["Prediction",       "K=6 multimodal trajectories, T=60 steps (6 seconds)"],
    ["Training",         "75 epochs, WTA + Huber loss, AMP fp16, RTX 4060 8GB"],
    ["Safety Classifier","RandomForest (300 trees), 18 kinematic features, 5 classes"],
    ["LLM Diagnosis",    "Llama-3.2-3B-Instruct, 4-bit NF4 quantization"],
]
cfg_table = Table(cfg_data, colWidths=[4.5*cm, 12*cm])
cfg_table.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0),  BG_HEADER),
    ("TEXTCOLOR",   (0,0), (-1,0),  CYAN),
    ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 8),
    ("FONTNAME",    (0,1), (0,-1),  "Helvetica-Bold"),
    ("TEXTCOLOR",   (0,1), (0,-1),  YELLOW),
    ("TEXTCOLOR",   (1,1), (1,-1),  WHITE),
    ("BACKGROUND",  (0,1), (-1,-1), BG_PANEL),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[BG_PANEL, BG_DARK]),
    ("GRID",        (0,0), (-1,-1), 0.3, GRAY),
    ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",  (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
]))
story.append(cfg_table)
story.append(Spacer(1, 0.4*cm))

# ── training history ──────────────────────────────────────────────────────────
story.append(section_header("Per-Epoch Training Performance"))

epoch_header = ["Epoch", "Train Loss", "val_ADE (m)", "val_FDE (m)"]
epoch_rows   = [epoch_header]
best_ade = min(h["minADE"] for h in history if h["minADE"])
for h in history:
    ade = h["minADE"] if h["minADE"] else float("nan")
    fde = h["minFDE"] if h["minFDE"] else float("nan")
    epoch_rows.append([
        str(h["epoch"]),
        f"{h['train_loss']:.4f}",
        f"{ade:.4f}",
        f"{fde:.4f}",
    ])

ep_table = Table(epoch_rows, colWidths=[2*cm, 3.5*cm, 3.5*cm, 3.5*cm])
ts = [
    ("BACKGROUND",   (0,0),  (-1,0),  BG_HEADER),
    ("TEXTCOLOR",    (0,0),  (-1,0),  CYAN),
    ("FONTNAME",     (0,0),  (-1,0),  "Helvetica-Bold"),
    ("FONTNAME",     (0,1),  (-1,-1), "Courier"),
    ("FONTSIZE",     (0,0),  (-1,-1), 8),
    ("TEXTCOLOR",    (1,1),  (1,-1),  ORANGE),
    ("TEXTCOLOR",    (2,1),  (2,-1),  GREEN),
    ("TEXTCOLOR",    (3,1),  (3,-1),  CYAN),
    ("TEXTCOLOR",    (0,1),  (0,-1),  WHITE),
    ("ALIGN",        (0,0),  (-1,-1), "CENTER"),
    ("GRID",         (0,0),  (-1,-1), 0.3, GRAY),
    ("TOPPADDING",   (0,0),  (-1,-1), 3),
    ("BOTTOMPADDING",(0,0),  (-1,-1), 3),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [BG_PANEL, BG_DARK]),
]
# highlight best epoch row
for i, h in enumerate(history, start=1):
    if h["minADE"] == best_ade:
        ts.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#1a3a1a")))
        ts.append(("TEXTCOLOR",  (2,i), (2,i),  GREEN))
        ts.append(("FONTNAME",   (0,i), (-1,i), "Courier-Bold"))
ep_table.setStyle(TableStyle(ts))
story.append(ep_table)
story.append(Spacer(1, 0.3*cm))

best_ep = next(h["epoch"] for h in history if h["minADE"] == best_ade)
story.append(Paragraph(
    f"Best model saved at epoch {best_ep}  |  val_ADE = {best_ade:.4f} m",
    ParagraphStyle("note", fontName="Helvetica-Oblique", fontSize=8,
                   textColor=GREEN, spaceAfter=6)
))

# ── final results ─────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(section_header("Final Test Results"))

res_data = [
    ["Metric",              "Value"],
    ["Test minADE",         f"{results['test_minADE']:.4f} m"],
    ["Test minFDE",         f"{results['test_minFDE']:.4f} m"],
    ["Best Val minADE",     f"{results['best_val_minADE']:.4f} m"],
    ["Macro Precision",     f"{prec:.4f}"],
    ["Macro Recall",        f"{rec:.4f}"],
    ["Macro F1",            f"{f1:.4f}"],
]
res_table = Table(res_data, colWidths=[5*cm, 4*cm])
res_table.setStyle(TableStyle([
    ("BACKGROUND",   (0,0),  (-1,0),  BG_HEADER),
    ("TEXTCOLOR",    (0,0),  (-1,0),  CYAN),
    ("FONTNAME",     (0,0),  (-1,0),  "Helvetica-Bold"),
    ("FONTNAME",     (0,1),  (0,-1),  "Helvetica-Bold"),
    ("FONTNAME",     (1,1),  (1,-1),  "Courier-Bold"),
    ("FONTSIZE",     (0,0),  (-1,-1), 9),
    ("TEXTCOLOR",    (0,1),  (0,-1),  YELLOW),
    ("TEXTCOLOR",    (1,1),  (1,-1),  GREEN),
    ("BACKGROUND",   (0,1),  (-1,-1), BG_PANEL),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [BG_PANEL, BG_DARK]),
    ("GRID",         (0,0),  (-1,-1), 0.3, GRAY),
    ("ALIGN",        (1,0),  (1,-1),  "CENTER"),
    ("TOPPADDING",   (0,0),  (-1,-1), 6),
    ("BOTTOMPADDING",(0,0),  (-1,-1), 6),
    ("LEFTPADDING",  (0,0),  (-1,-1), 10),
]))
story.append(res_table)
story.append(Spacer(1, 0.3*cm))

# Final Results line (friend style)
story.append(Paragraph("Terminal Output", S_H2))
final_line = (f"Final Results: {{'ADE': {results['test_minADE']:.4f}, "
              f"'FDE': {results['test_minFDE']:.4f}, "
              f"'precision': {prec:.2f}, 'recall': {rec:.2f}, 'f1': {f1:.2f}}}")
story.append(Paragraph(final_line, S_MONO))
story.append(Spacer(1, 0.4*cm))

# ── per-class safety table ─────────────────────────────────────────────────────
story.append(section_header("Safety Classification Report (Test Set)"))

class_data = [["Class", "Precision", "Recall", "F1", "Support"]]
class_lines = [l for l in results["test_safety_report"].splitlines()
               if l.strip() and "accuracy" not in l and "avg" not in l
               and "precision" not in l]
for line in class_lines:
    parts = line.split()
    if len(parts) >= 5:
        name    = " ".join(parts[:-4])
        p,r,f,s = parts[-4], parts[-3], parts[-2], parts[-1]
        class_data.append([name, p, r, f, s])

cls_table = Table(class_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 2*cm, 2*cm])
cls_ts = [
    ("BACKGROUND",   (0,0),  (-1,0),  BG_HEADER),
    ("TEXTCOLOR",    (0,0),  (-1,0),  CYAN),
    ("FONTNAME",     (0,0),  (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",     (0,0),  (-1,-1), 8.5),
    ("TEXTCOLOR",    (0,1),  (0,-1),  WHITE),
    ("FONTNAME",     (0,1),  (0,-1),  "Helvetica-Bold"),
    ("FONTNAME",     (1,1),  (-1,-1), "Courier"),
    ("TEXTCOLOR",    (1,1),  (-1,-1), GREEN),
    ("ALIGN",        (1,0),  (-1,-1), "CENTER"),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [BG_PANEL, BG_DARK]),
    ("GRID",         (0,0),  (-1,-1), 0.3, GRAY),
    ("TOPPADDING",   (0,0),  (-1,-1), 5),
    ("BOTTOMPADDING",(0,0),  (-1,-1), 5),
    ("LEFTPADDING",  (0,0),  (-1,-1), 8),
]
cls_table.setStyle(TableStyle(cls_ts))
story.append(cls_table)

# ── demo pipeline output ──────────────────────────────────────────────────────
story.append(PageBreak())
story.append(section_header("Demo Pipeline Output"))
story.append(Paragraph(
    "python src/demo_pipeline.py", S_MONO))
story.append(Spacer(1, 0.3*cm))

for entry in demo:
    label     = entry["event_label"]
    name      = entry["event_name"]
    features  = entry["features"]
    report    = entry["report"]
    ev_color  = EVENT_COLORS.get(name, WHITE)

    # sample header
    story.append(Paragraph(
        f"Predicted safety label: {label} ({name})",
        ParagraphStyle("evlabel", fontName="Courier-Bold", fontSize=9,
                       textColor=ev_color, backColor=BG_DARK,
                       leftIndent=6, spaceAfter=3, spaceBefore=6)
    ))

    # retrieved cases
    story.append(Paragraph("Retrieved cases:", S_H2))
    retrieved = retrieve(features)
    for sim, text in retrieved:
        story.append(Paragraph(
            f"&nbsp;&nbsp;{sim:.4f} -- {text}",
            ParagraphStyle("case", fontName="Courier", fontSize=8,
                           textColor=CYAN, backColor=BG_DARK,
                           leftIndent=6, leading=11, spaceAfter=2)
        ))

    # diagnosis
    story.append(Paragraph("Diagnosis:", S_H2))
    sev    = report.get("severity", "N/A")
    action = report.get("recommended_action", "")
    texts  = " | ".join(t for _, t in retrieved)
    diag   = (f"Predicted safety event detected. Similar prior cases suggest: {texts}. "
              f"Severity: {sev}. Recommended action: {action}")
    story.append(Paragraph(diag, S_DIAG))
    story.append(hr())

# ── build PDF ─────────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG_DARK)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(A4[0]/2, 1.2*cm,
        f"AV Safety Forecasting  |  DATA 612  |  Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF saved: {pdf_path}")
