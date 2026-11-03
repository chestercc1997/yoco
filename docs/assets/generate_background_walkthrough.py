"""Rebuild the architecture and inference walkthroughs from editable vector primitives."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yoco-walkthrough-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/yoco-walkthrough-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
BG = "#F7F9FC"
INK = "#192E45"
MUTED = "#52667D"
BLUE = "#285AB0"
TEAL = "#087F83"
PURPLE = "#7750AA"
ORANGE = "#B66616"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, s, size=16, color=INK, bold=False, ha="left"):
    return ax.text(x, y, s, fontsize=size, color=color,
                   weight="bold" if bold else "normal", ha=ha, va="top",
                   linespacing=1.5)


def card(ax, x, y, w, h, title, body="", color=BLUE, fill="white", size=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=10", facecolor=fill,
        edgecolor=color, linewidth=1.2))
    t = label(ax, x + 18, y + 13, title, 17, color, True)
    t._card_bounds = (x + 12, y + 8, w - 24, h - 16)
    if body:
        b = label(ax, x + 18, y + 43, body, size)
        b._card_bounds = (x + 12, y + 8, w - 24, h - 16)


def arrow(ax, points, color=MUTED, dashed=False):
    style = "--" if dashed else "-"
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, lw=1.8, ls=style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
        mutation_scale=15, linewidth=1.8, color=color, linestyle=style))


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for artist in ax.texts:
            if hasattr(artist, "_card_bounds"):
                x, y, w, h = artist._card_bounds
                p0, p1 = ax.transData.transform([(x, y), (x + w, y + h)])
                b = artist.get_window_extent(renderer)
                available = p1[0] - b.x0
                if b.width > available:
                    artist.set_fontsize(artist.get_fontsize() * available / b.width * .98)
    fig.canvas.draw()
    for ax in fig.axes:
        for artist in ax.texts:
            if not hasattr(artist, "_card_bounds"):
                continue
            x, y, w, h = artist._card_bounds
            p0, p1 = ax.transData.transform([(x, y), (x + w, y + h)])
            b = artist.get_window_extent(renderer)
            assert b.x0 >= p0[0] - 1 and b.x1 <= p1[0] + 1, artist.get_text()
            assert b.y0 >= p1[1] - 1 and b.y1 <= p0[1] + 1, artist.get_text()
    fig.savefig(OUT / (name + ".svg"), facecolor=BG)
    fig.savefig(OUT / (name + ".png"), dpi=125, facecolor=BG)
    plt.close(fig)


fig, ax = canvas(1720, 1540)
label(ax, 50, 28, "From token IDs to next-token logits", 28, bold=True)
label(ax, 50, 77, "Full forward dependencies  |  read each lane top to bottom; inference scheduling is shown separately", 16, MUTED)
label(ax, 60, 125, "A   CAUSAL TRANSFORMER", 21, BLUE, True)
label(ax, 790, 125, "B   YOCO-gRet 160M + optional depth fusion", 21, TEAL, True)

# The conventional block is expanded in exact computation order.
card(ax, 60, 175, 600, 88, "Token IDs → embedding", "x[1:T] → X0 [B,T,d]", BLUE)
ax.add_patch(FancyBboxPatch((45, 295), 630, 1000,
    boxstyle="round,pad=0,rounding_size=14", facecolor="#EDF3FC", edgecolor="#B7C9E6"))
label(ax, 65, 307, "BLOCK l   ·   repeat for l = 1,…,L", 16, BLUE, True)
steps = [
    (350, 85, "1   RMSNorm", "Input: previous block output X(l−1)"),
    (465, 100, "2   Q = U Wq   |   K = U Wk   |   V = U Wv", "Split into heads; RoPE on Q and K"),
    (595, 90, "3   Attention scores", "score = Q Kᵀ / √d_head + causal mask"),
    (715, 90, "4   Normalize scores, then read values", "A = softmax(score);  O_head = A V"),
    (835, 100, "5   Merge heads → output projection → add", "R = X(l−1) + ConcatHeads(O_head) Wo"),
    (965, 110, "6   RMSNorm → SwiGLU FFN", "F = Norm(R)\nD = [SiLU(F W1) ⊙ (F W3)] W2"),
    (1105, 90, "7   Add residual → next block", "X(l) = R + D"),
]
for y, h, title, body in steps:
    card(ax, 70, y, 580, h, title, body, BLUE)
arrow(ax, [(360, 263), (360, 280), (640, 280), (640, 350)])
for a, b in [(435, 465), (565, 595), (685, 715), (805, 835), (935, 965), (1075, 1105)]:
    arrow(ax, [(360, a), (360, b)])
label(ax, 75, 1220, "Each layer has its own Q/K/V projections and KV cache.", 14, BLUE)
label(ax, 75, 1250, "The FFN output becomes the next layer's input.", 14, BLUE)
card(ax, 60, 1340, 600, 105, "Final RMSNorm → LM head → token selection", "Last-position logits → next token ID\nNext generation step: look up that ID's embedding", BLUE)
arrow(ax, [(360, 1195), (360, 1208), (690, 1208), (690, 1315), (360, 1315), (360, 1340)])

# YOCO main stream, with a separate memory stream and optional fusion branch.
card(ax, 790, 175, 530, 88, "Token IDs → scaled embedding", "X0 = √768 × Embedding(x[1:T])", TEAL)
card(ax, 790, 310, 530, 270, "SELF BLOCK × 6   ·   each has its own state", "RMSNorm → Q, K, V + gates\nRoPE(Q,K) → update recurrent state S\nRead qS → head norm / gate → Wout\nAdd residual\nRMSNorm → SwiGLU FFN → add residual", TEAL, fill="#F0F9F8", size=15)
arrow(ax, [(1055, 263), (1055, 310)], TEAL)
card(ax, 790, 625, 530, 90, "H6: final self-block output", "One vector per token; shape [B,T,768]", TEAL)
arrow(ax, [(1055, 580), (1055, 625)], TEAL)

card(ax, 1390, 310, 285, 225, "OPTIONAL: dynamic v2", "Tap H2, H4, H6\nfrom self-block outputs.\nRouter(H6) → α\nM = α2 H2 + α4 H4\n       + α6 H6", ORANGE, fill="#FFF5E8", size=13)
arrow(ax, [(1320, 445), (1390, 445)], ORANGE, True)
card(ax, 1390, 625, 285, 125, "Memory input M", "Baseline: M = H6\nDynamic: fused M above", PURPLE, fill="#F5F0FA", size=13)
arrow(ax, [(1320, 672), (1390, 672)], PURPLE)
arrow(ax, [(1532, 535), (1532, 625)], ORANGE, True)
card(ax, 1390, 810, 285, 180, "Shared K and V", "Norm(M) → Wk, Wv\nRoPE on K\n[B,T,12,64] each\nOne pair for all 6\ncross-decoder blocks", PURPLE, fill="#F5F0FA", size=13)
arrow(ax, [(1532, 750), (1532, 810)], PURPLE)

card(ax, 790, 810, 530, 385, "CROSS BLOCK × 6   ·   each has its own Q", "Input C0 = H6; then previous cross output\nRMSNorm → Wq → RoPE → Q_j\nQ_j Kᵀ / √64 + causal mask → scores\nsoftmax(scores) → A_j → A_j V\nMerge heads → Wo,j → add residual\nRMSNorm → SwiGLU FFN → add residual\nOutput C_j → next cross block", PURPLE, fill="#F5F0FA", size=15)
arrow(ax, [(1055, 715), (1055, 810)], TEAL)
arrow(ax, [(1390, 916), (1320, 916)], PURPLE)
label(ax, 1360, 1013, "Same K/V; new scores\nand new FFN at each layer.", 13, PURPLE)
card(ax, 790, 1340, 530, 105, "Final RMSNorm → tied LM head", "Last-position logits → next token ID\nTraining: next-token loss at every valid position", TEAL, size=14)
arrow(ax, [(1055, 1195), (1055, 1340)], PURPLE)
label(ax, 1390, 1110, "Cache = 6 retention states\n+ 1 global KV pair.\nGlobal KV history grows\nwith context length.", 13, TEAL)
label(ax, 60, 1483, "Blue / teal: hidden-state path    Purple: shared memory    Orange dashed: optional fusion    ×N: sequential blocks", 15, MUTED)
save(fig, "background_architecture_overview")


fig, ax = canvas(1620, 1060)
label(ax, 45, 28, "YOCO generation: prefill → first output → subsequent decode", 26, bold=True)
label(ax, 45, 77, "Local benchmark convention: prefill the first T−1 prompt tokens; process the final prompt token in the decode loop.", 14, MUTED)
columns = [(45, BLUE), (585, TEAL), (1125, PURPLE)]
headings = ["A   PREFIX PREFILL", "B   FIRST OUTPUT", "C   NEXT DECODE STEP"]
rows = [
    [
        ("Known prefix: x[1:T−1]", "Many input token IDs → embeddings"),
        ("Self blocks × 6", "Retention + FFN at every prefix position\nSave each layer's final state S_l,T−1"),
        ("Construct shared prefix memory", "H6 → M → Norm → K,V projections\nSave K[1:T−1], V[1:T−1]"),
        ("Skip cross blocks for this prefix", "Their historical outputs are not needed\nto construct future queries or memory."),
        ("Ready to process the prompt's last token", "No usable generated token yet.\nKeep all six states and the shared KV."),
    ],
    [
        ("Last prompt token: x_T", "One input token ID → embedding"),
        ("Self blocks × 6", "Read S_l,T−1 → update to S_l,T\nRetention + residual + FFN + residual"),
        ("Append this position's shared K,V", "H6_T → M_T → k_T,v_T\nGlobal memory now covers 1:T"),
        ("Cross blocks × 6", "Q → scores over K[1:T] → softmax → AV\nOutput projection / residual → FFN / residual"),
        ("Final norm → head → select x_(T+1)", "This is the FIRST generated token.\nThe input to this phase was still x_T."),
    ],
    [
        ("Just-generated token: x_(T+1)", "One input token ID → embedding"),
        ("Self blocks × 6", "Read S_l,T → update to S_l,T+1\nRetention + residual + FFN + residual"),
        ("Append this position's shared K,V", "H6_(T+1) → M_(T+1) → k_(T+1),v_(T+1)\nGlobal memory now covers 1:T+1"),
        ("Cross blocks × 6", "Current Q reads the updated shared memory\nAll six attention and FFN blocks execute."),
        ("Final norm → head → select x_(T+2)", "Feed this token back for the next step.\nRepeat until the stopping condition."),
    ],
]
ys = [170, 335, 500, 665, 830]
for i, (x, color) in enumerate(columns):
    label(ax, x, 126, headings[i], 19, color, True)
    for y, (title, body) in zip(ys, rows[i]):
        card(ax, x, y, 450, 125, title, body, color, size=12.5)
    for y, next_y in zip(ys, ys[1:]):
        arrow(ax, [(x + 225, y + 125), (x + 225, next_y)], color)
# Carry the layer states and global memory along distinct arrows.
for right, left, color in [(495, 585, BLUE), (1035, 1125, TEAL)]:
    arrow(ax, [(right, 400), (left, 400)], color, True)
    label(ax, right + 14, 376, "6 states", 9, color)
    arrow(ax, [(right, 563), (left, 563)], color, True)
    label(ax, right + 20, 539, "K / V", 10, color)
arrow(ax, [(1035, 900), (1100, 900), (1100, 155), (1350, 155), (1350, 170)], TEAL)
arrow(ax, [(1575, 900), (1600, 900), (1600, 230), (1575, 230)], PURPLE)
label(ax, 45, 996, "M = H6 in baseline; M = depth fusion in dynamic v2.  The query stream always starts from H6.", 15, MUTED)
save(fig, "background_yoco_prefill_decode")
