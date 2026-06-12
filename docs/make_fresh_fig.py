"""Paper-ready architecture figure for `fresh_extract` (run 11_fresh_f1).
Faithful to core/model.py: FreshExtract, _ModalityStem, _DeepEncoder (CHANNELS
[96,160,256,384,512] @ [256,128,64,32,16]), per-level 1x1 fusion, two _PyrDecoder,
heads frac(4)/binary(1)/height(1). TerraMind path shown dashed (OFF in 11_fresh_f1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

C_ALPHA = "#2E6F9E"; C_TESS = "#3B8C5A"; C_FUSE = "#9B59B6"; C_PYR = "#5D6D7E"
C_FDEC = "#C0712B"; C_HDEC = "#B03A48"; C_OFF = "#AAB2BD"; EDGE = "#2C3E50"

fig, ax = plt.subplots(figsize=(17, 9.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, label, fc, fs=10, tc="white", bold=True, alpha=1.0, ls="-", ec=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.0",
                 linewidth=1.4, edgecolor=ec, facecolor=fc, alpha=alpha, linestyle=ls))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=5)
    return (x, y, w, h)

def arrow(p0, p1, color=EDGE, lw=1.6, ls="-", rad=0.0, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=lw,
                 color=color, linestyle=ls, alpha=alpha,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))

def rc(b): return (b[0]+b[2], b[1]+b[3]/2)
def lc(b): return (b[0], b[1]+b[3]/2)
def tc_(b): return (b[0]+b[2]/2, b[1]+b[3])
def bc(b): return (b[0]+b[2]/2, b[1])

# ===== Title =====
ax.text(50, 54.6, "fresh_extract  —  single self-contained model  (run 11_fresh_f1)",
        ha="center", fontsize=15.5, fontweight="bold", color="#1B2631")
ax.text(50, 52.3, "Deep dual pixel encoders  +  symmetric per-level fusion  →  shared pyramid  →  two decoders"
        "   (~50M params, bf16-AMP)", ha="center", fontsize=10.5, color="#566573")

# ===== Inputs =====
ai = box(1.0, 38.5, 12.5, 6.2, "AlphaEarth\n64 ch @ 256x256\n(S2/S1/PALSAR,\nGEDI-LiDAR, DEM)", C_ALPHA, fs=8.4)
ti = box(1.0, 7.0, 12.5, 6.2, "TESSERA\n128 ch @ 256x256\n(1 yr S1+S2\ntime-series)", C_TESS, fs=8.4)

# ===== Stems =====
asx = box(16, 39.6, 9, 4.0, "Alpha Stem\n→ 96 @ 256", C_ALPHA, fs=8.6)
tsx = box(16, 8.1, 9, 4.0, "Tessera Stem\n→ 96 @ 256", C_TESS, fs=8.6)
arrow(rc(ai), lc(asx), C_ALPHA); arrow(rc(ti), lc(tsx), C_TESS)

# ===== Encoders as 5-level ladders =====
LV = [("96","256"), ("160","128"), ("256","64"), ("384","32"), ("512","16")]
enc_w = 21.0; enc_x = 27.5; bh = 3.0; sp = 3.85
def ladder(y0, color, title):
    boxes = []
    for i,(ch,res) in enumerate(LV):
        bw = 8.5 + i*2.2
        bx = enc_x + (enc_w-bw)/2
        by = y0 + i*sp
        boxes.append(box(bx, by, bw, bh, f"L{i}: {ch}@{res}", color, fs=7.4))
    top = y0 + 4*sp + bh
    ax.text(enc_x+enc_w/2, top+1.1, title, ha="center", fontsize=9.0, fontweight="bold", color=color)
    return boxes
aenc = ladder(30.0, C_ALPHA, "Alpha Encoder (5 levels)")
tenc = ladder(1.0, C_TESS, "Tessera Encoder (5 levels)")
arrow(rc(asx), lc(aenc[0]), C_ALPHA); arrow(rc(tsx), lc(tenc[0]), C_TESS)
for i in range(4):
    arrow(tc_(aenc[i]), bc(aenc[i+1]), C_ALPHA, lw=0.9)
    arrow(tc_(tenc[i]), bc(tenc[i+1]), C_TESS, lw=0.9)

# ===== Per-level symmetric fusion -> shared pyramid =====
fus_x = 53.0; pyr_x = 60.0; pyr_w = 12.0
pyr_boxes = []
for i,(ch,res) in enumerate(LV):
    cy = (aenc[i][1]+bh/2 + tenc[i][1]+bh/2)/2
    ax.add_patch(Circle((fus_x, cy), 1.45, facecolor=C_FUSE, edgecolor=EDGE, lw=1.2, zorder=4))
    ax.text(fus_x, cy, "1x1", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold", zorder=5)
    arrow(rc(aenc[i]), (fus_x-1.45, cy), C_ALPHA, lw=1.0, rad=-0.12)
    arrow(rc(tenc[i]), (fus_x-1.45, cy), C_TESS, lw=1.0, rad=0.12)
    pb = box(pyr_x, cy-1.55, pyr_w, 3.1, f"{ch}@{res}", C_PYR, fs=7.6)
    pyr_boxes.append(pb)
    arrow((fus_x+1.45, cy), lc(pb), C_FUSE, lw=1.1)
ax.text(pyr_x+pyr_w/2, pyr_boxes[-1][1]+pyr_boxes[-1][3]+1.3, "Shared Feature Pyramid",
        ha="center", fontsize=9.0, fontweight="bold", color=C_PYR)
ax.text(fus_x, pyr_boxes[-1][1]+pyr_boxes[-1][3]+1.3, "concat\n→1x1", ha="center", va="bottom",
        fontsize=6.8, color=C_FUSE, fontweight="bold")

# group rectangle around pyramid (so we draw ONE arrow to each decoder, not 10)
gy0 = pyr_boxes[0][1]-1.0; gy1 = pyr_boxes[-1][1]+pyr_boxes[-1][3]+0.6
ax.add_patch(FancyBboxPatch((pyr_x-1.0, gy0), pyr_w+2.0, gy1-gy0,
             boxstyle="round,pad=0.1,rounding_size=1.0", linewidth=1.3,
             edgecolor=C_PYR, facecolor="none", linestyle=(0,(3,2)), zorder=1))

# ===== TerraMind (OFF) =====
tm = box(54.0, 47.4, 17, 4.4, "TerraMind tokens 768@16x16\n(Bet 3 — OFF in 11_fresh_f1)",
         C_OFF, fs=7.4, tc="#5D6D7E", ls=(0,(4,3)), ec=C_OFF)
arrow(bc(tm), tc_(pyr_boxes[4]), C_OFF, lw=1.0, ls=(0,(4,3)))

# ===== Two decoders (read the WHOLE pyramid group) =====
fdec = box(78.0, 35.0, 11.0, 7.0, "Fraction\nDecoder\n(16→256)", C_FDEC, fs=9)
hdec = box(78.0, 6.5, 11.0, 7.0, "Height\nDecoder\n(16→256)", C_HDEC, fs=9)
gx = pyr_x+pyr_w+1.0; gmid = (gy0+gy1)/2
arrow((gx, gmid), lc(fdec), C_FDEC, lw=2.2, rad=-0.18)
arrow((gx, gmid), lc(hdec), C_HDEC, lw=2.2, rad=0.18)
ax.text(73.0, gmid+5.0, "all 5\nlevels", ha="center", fontsize=7.0, color="#566573", style="italic")

# ===== Heads =====
fh = box(91.0, 38.8, 8.0, 3.4, "frac_head\n4ch", C_FDEC, fs=7.6)
bh2 = box(91.0, 33.6, 8.0, 3.4, "binary_head\n1ch", C_FDEC, fs=7.6)
hh = box(91.0, 8.3, 8.0, 3.4, "height_head\n1ch", C_HDEC, fs=7.6)
arrow(rc(fdec), lc(fh), C_FDEC, rad=0.08); arrow(rc(fdec), lc(bh2), C_FDEC, rad=-0.08)
arrow(rc(hdec), lc(hh), C_HDEC)

# ===== Output text =====
ax.text(99.6, 40.5, "softmax4\nB / V / W\n(+'other')", ha="left", va="center", fontsize=7.6, color=C_FDEC, fontweight="bold")
ax.text(99.6, 35.3, "aux building\n(train only)", ha="left", va="center", fontsize=7.0, color="#7F8C8D")
ax.text(99.6, 10.0, "height (m)\n×30 denorm", ha="left", va="center", fontsize=7.6, color=C_HDEC, fontweight="bold")

# ===== Bet-1 note (clean, bottom-right, no crossing arrows) =====
ax.text(83.5, 2.6, "Bet 1: the HEIGHT decoder reads the shared pyramid →\n"
        "AlphaEarth (GEDI-LiDAR / DEM) reaches height FULLY — no throttled bridge",
        ha="center", va="center", fontsize=7.6, color="#1B2631",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FDF6E3", ec=C_HDEC, lw=1.0))

# ===== Legend =====
leg = [
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_ALPHA, markersize=11, label='AlphaEarth path'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_TESS, markersize=11, label='TESSERA path'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_FUSE, markersize=11, label='per-level 1x1 fusion'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_PYR, markersize=11, label='shared pyramid'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_FDEC, markersize=11, label='fraction decoder/heads'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_HDEC, markersize=11, label='height decoder/head'),
    Line2D([0],[0], color=C_OFF, lw=1.6, ls=(0,(4,3)), label='TerraMind (disabled)'),
]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.003, 0.045), ncol=4,
          frameon=True, fontsize=8.0, handletextpad=0.5, columnspacing=1.2)

ax.text(50, 0.6,
        "Objective (sm4tv):  fraction = softmax4 + Tversky(a=0.3,b=0.7,g=1.33) + soft-Dice(k=5)   |   "
        "height = masked Huber(d=0.5)   |   binary = BCE+Dice    .    task weights f/h/b = 0.65 / 0.64 / 1.70",
        ha="center", fontsize=8.0, color="#566573")

plt.tight_layout()
fig.savefig("docs/fresh_extract_arch.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("docs/fresh_extract_arch.pdf", bbox_inches="tight", facecolor="white")
print("saved")
