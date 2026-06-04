"""Figure builders for the four distillation methods.

Pure matplotlib — each function takes the same control values the marimo
notebook exposes as sliders, and returns a ``matplotlib.figure.Figure``. Both
``distillation_viz.py`` (interactive) and ``export_figures.py`` (static PNGs for
the README) import from here, so the picture is defined exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Shared palette — see the legend at the top of the notebook / README.
TEACHER_C = "#2f6db5"  # blue  — teacher / target
STUDENT_C = "#d1772e"  # orange — student / what we train
KEPT_C = "#c2403d"  # red    — signal a method keeps
MUTED_C = "#c9ccd1"  # grey   — signal thrown away

# Toy next-token distribution after a partial answer like "...so the total is ".
TOY_VOCAB = ["12", "twelve", "10", "13", "11", "14", "a", "$"]
_TOY_TEACHER = np.array([0.55, 0.20, 0.10, 0.06, 0.04, 0.03, 0.012, 0.008])
TOY_TEACHER_PROBS = _TOY_TEACHER / _TOY_TEACHER.sum()


def _despine(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Method 1 · Sequence-level
# --------------------------------------------------------------------------- #
def sequence_level(confidence: float = 0.55):
    """Teacher's full distribution vs the one-hot label the student is taught."""
    base = np.array([0.0, 0.20, 0.10, 0.06, 0.04, 0.03, 0.012, 0.008])
    base = base / base.sum()
    p = np.concatenate([[confidence], base[1:] * (1 - confidence)])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    x = np.arange(len(TOY_VOCAB))

    axL.bar(x, p, color=TEACHER_C)
    axL.set_title("What the teacher knows\n(full next-token distribution)", fontsize=10)
    axL.set_ylabel("probability")

    argmax = int(np.argmax(p))
    onehot = np.zeros_like(p)
    onehot[argmax] = 1.0
    colors = [KEPT_C if i == argmax else MUTED_C for i in range(len(p))]
    axR.bar(x, onehot, color=colors)
    axR.set_title("What the student is taught\n(one-hot — the rest is discarded)", fontsize=10)

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(TOY_VOCAB, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        _despine(ax)
    axR.text(
        0.97,
        0.9,
        f"{1 - confidence:.0%} of the\nteacher's mass\nthrown away",
        transform=axR.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=KEPT_C,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Method 2 · Token-level KL
# --------------------------------------------------------------------------- #
def _soften(p, T):
    logits = np.log(p + 1e-12) / T
    e = np.exp(logits - logits.max())
    return e / e.sum()


def token_level_match(T: float = 2.0):
    """Teacher vs a half-trained student distribution at one position, at temp T."""
    teach = _soften(TOY_TEACHER_PROBS, T)
    student_raw = np.array([0.40, 0.08, 0.18, 0.05, 0.05, 0.04, 0.12, 0.08])
    student_raw = student_raw / student_raw.sum()
    stud = _soften(student_raw, T)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(TOY_VOCAB))
    w = 0.4
    ax.bar(x - w / 2, teach, w, label="teacher", color=TEACHER_C)
    ax.bar(x + w / 2, stud, w, label="student", color=STUDENT_C)

    kl = float((teach * (np.log(teach + 1e-12) - np.log(stud + 1e-12))).sum())
    ax.set_title(
        f"Matching the full distribution at one position   ·   KL(teacher‖student) = {kl:.3f}",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TOY_VOCAB, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("probability")
    ax.legend(frameon=False)
    _despine(ax)
    ax.text(
        0.98,
        0.95,
        f"higher T flattens both bars,\nrevealing the ranking of rare tokens\n(T = {T:g})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555",
    )
    fig.tight_layout()
    return fig


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def kl_direction(reverse: bool = False):
    """Best single-mode student fit to a bimodal teacher, under forward/reverse KL."""
    x = np.linspace(-6, 8, 600)
    teacher = 0.5 * _gauss(x, -2.0, 0.7) + 0.5 * _gauss(x, 3.5, 0.9)
    teacher = teacher / np.trapezoid(teacher, x)

    best = None
    for mu in np.linspace(-3, 5, 90):
        for sig in np.linspace(0.4, 4.0, 90):
            q = _gauss(x, mu, sig)
            q = q / np.trapezoid(q, x)
            if reverse:  # KL(student || teacher)
                kl = np.trapezoid(q * (np.log(q + 1e-12) - np.log(teacher + 1e-12)), x)
            else:  # KL(teacher || student)
                kl = np.trapezoid(teacher * (np.log(teacher + 1e-12) - np.log(q + 1e-12)), x)
            if best is None or kl < best[0]:
                best = (kl, mu, sig)

    q = _gauss(x, best[1], best[2])
    q = q / np.trapezoid(q, x)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.fill_between(x, teacher, color=TEACHER_C, alpha=0.25)
    ax.plot(x, teacher, color=TEACHER_C, lw=2, label="teacher (two modes)")
    ax.plot(x, q, color=STUDENT_C, lw=2.4, label="best single-mode student")
    label = (
        "reverse KL → locks onto ONE mode (sharp, may ignore the other)"
        if reverse
        else "forward KL → stretches to COVER both modes (hedges the middle)"
    )
    ax.set_title(label, fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("a behaviour the model could produce")
    ax.legend(frameon=False, loc="upper center")
    _despine(ax, ("top", "right", "left"))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Method 3 · On-policy (GKD)
# --------------------------------------------------------------------------- #
def on_policy(alpha: float = 0.5):
    """Left: the on/off-policy step mix. Right: exposure bias as distribution shift."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})

    axL.barh([0], [alpha], color=STUDENT_C, height=0.5)
    axL.barh([0], [1 - alpha], left=[alpha], color=TEACHER_C, height=0.5)
    axL.set_xlim(0, 1)
    axL.set_ylim(-1, 1)
    axL.set_yticks([])
    axL.set_xlabel("fraction of training steps")
    axL.set_title("Where each step's prefix comes from", fontsize=10)
    if alpha > 0.07:
        axL.text(alpha / 2, 0, f"student\nrollout\n{alpha:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if alpha < 0.93:
        axL.text(
            alpha + (1 - alpha) / 2,
            0,
            f"teacher\ntext\n{1 - alpha:.0%}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    axL.text(
        0.5,
        -0.7,
        "pure on-policy (α=1) can diverge early — the\nGKD mix keeps it stable",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555",
    )
    _despine(axL, ("top", "right", "left"))

    x = np.linspace(-5, 9, 500)

    def g(mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    teacher_dist = g(1.0, 1.0)
    infer = g(4.5, 1.4)  # where the student actually generates at test time
    train_mu = 1.0 + alpha * 3.5
    train_s = 1.0 + alpha * 0.4
    train = g(train_mu, train_s)

    axR.plot(x, teacher_dist, color=TEACHER_C, lw=2, label="teacher's prefixes (train, α=0)")
    axR.plot(x, infer, color="#888", lw=2, ls="--", label="student at inference (test)")
    axR.fill_between(x, train, color=STUDENT_C, alpha=0.3)
    axR.plot(x, train, color=STUDENT_C, lw=2.2, label=f"actual training dist. (α={alpha:.1f})")
    axR.set_title(f"Exposure-bias gap (train vs test): {abs(4.5 - train_mu):.1f}", fontsize=10)
    axR.set_yticks([])
    axR.set_xlabel("prefix distribution the student is trained on")
    axR.legend(frameon=False, fontsize=7.5, loc="upper right")
    _despine(axR, ("top", "right", "left"))

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Method 4 · Cross-tokenizer (ULD)
# --------------------------------------------------------------------------- #
def uld_alignment():
    """The same text segmented two ways, aligned by ending character (as in the repo)."""
    text = "He has 12 apples."
    student_toks = [("He", 0, 2), (" has", 2, 6), (" 12", 6, 9), (" apples", 9, 16), (".", 16, 17)]
    teacher_toks = [("He", 0, 2), (" has", 2, 6), (" 1", 6, 8), ("2", 8, 9), (" app", 9, 13), ("les", 13, 16), (".", 16, 17)]
    teacher_ends = [e for _, _, e in teacher_toks]

    def nearest(s_end):
        best, bd = 0, 1e9
        for j, e in enumerate(teacher_ends):
            if abs(e - s_end) < bd:
                best, bd = j, abs(e - s_end)
        return best

    fig, ax = plt.subplots(figsize=(10, 3.6))
    scale = 0.62

    def draw_row(toks, y, color, label):
        for tok, s, e in toks:
            ax.add_patch(plt.Rectangle((s * scale, y), (e - s) * scale, 0.7, facecolor=color, edgecolor="white", lw=1.5, alpha=0.85))
            ax.text((s + e) / 2 * scale, y + 0.35, tok, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(-0.4, y + 0.35, label, ha="right", va="center", fontsize=9, color=color, fontweight="bold")

    draw_row(teacher_toks, 2.0, TEACHER_C, "teacher\n(SmolLM2)")
    draw_row(student_toks, 0.0, STUDENT_C, "student\n(Qwen)")

    for _tok, s, e in student_toks:
        _, t_start, t_end = teacher_toks[nearest(e)]
        x0 = (s + e) / 2 * scale
        x1 = (t_start + t_end) / 2 * scale
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 0.7), arrowprops={"arrowstyle": "->", "color": "#999", "lw": 1.3})

    ax.set_xlim(-2.2, 18 * scale)
    ax.set_ylim(-0.6, 3.1)
    ax.axis("off")
    ax.set_title(f'Same text "{text}", two segmentations, aligned by ending character', fontsize=10)
    ax.text(9 * scale, 2.95, 'teacher\'s " 1"/"2" split gets collapsed — alignment is approximate', ha="center", fontsize=8.5, color="#666")
    fig.tight_layout()
    return fig


def uld_sorted_topk(top_k: int = 6):
    """Sorted top-K probability shapes, teacher vs student — token identity discarded."""
    rng_t = np.array([0.46, 0.21, 0.12, 0.07, 0.05, 0.035, 0.025, 0.015, 0.01, 0.005, 0.003, 0.002])
    rng_s = np.array([0.38, 0.25, 0.10, 0.09, 0.06, 0.04, 0.03, 0.02, 0.012, 0.008, 0.005, 0.003])
    t_top = np.sort(rng_t)[::-1][:top_k]
    s_top = np.sort(rng_s)[::-1][:top_k]
    t_top = t_top / t_top.sum()
    s_top = s_top / s_top.sum()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    xt = np.arange(top_k)
    axL.bar(xt, t_top, color=TEACHER_C)
    axL.set_title(f"teacher: top-{top_k} probs, SORTED\n(which tokens? — discarded)", fontsize=10)
    axL.set_xlabel("rank")
    axL.set_ylabel("renormalised probability")
    axR.bar(xt, s_top, color=STUDENT_C)
    axR.set_title(f"student: top-{top_k} probs, SORTED\nKL compares these two shapes", fontsize=10)
    axR.set_xlabel("rank")
    for ax in (axL, axR):
        ax.set_xticks(xt)
        _despine(ax)
    kl = float((t_top * (np.log(t_top + 1e-12) - np.log(s_top + 1e-12))).sum())
    axR.text(
        0.97, 0.92, f"sorted-KL = {kl:.3f}\ntail past K is gone", transform=axR.transAxes, ha="right", va="top", fontsize=9, color="#555"
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Epilogue · real logged loss curves
# --------------------------------------------------------------------------- #
def _find_logs(start=None):
    bases = [Path.cwd(), Path.cwd().parent]
    if start is not None:
        bases.append(Path(start).resolve().parent.parent)
    for base in bases:
        cand = base / "logs"
        if cand.is_dir():
            return cand
    return None


def loss_curves(logs_dir=None):
    """The actual logged loss trajectories for all five runs (2×3 grid)."""
    logs = Path(logs_dir) if logs_dir else _find_logs(__file__)

    def load(name, key):
        if logs is None:
            return [], []
        p = logs / name
        if not p.exists():
            return [], []
        xs, ys = [], []
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                if key in r:
                    xs.append(r.get("step", len(xs)))
                    ys.append(r[key])
        return xs, ys

    panels = [
        ("Sequence-level (CE)", "sequence_level.jsonl", "loss", STUDENT_C),
        ("Token-level forward KL", "logits_forward.jsonl", "kl", TEACHER_C),
        ("Token-level reverse KL", "logits_reverse.jsonl", "kl", TEACHER_C),
        ("On-policy GKD (reverse KL)", "onpolicy.jsonl", "loss", KEPT_C),
        ("Cross-tokenizer ULD", "uld.jsonl", "loss", "#6a3d9a"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    axes = axes.ravel()
    for ax, (title, file, key, c) in zip(axes, panels, strict=False):
        xs, ys = load(file, key)
        if xs:
            ax.plot(xs, ys, color=c, lw=1.4)
            ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color=c, s=18, zorder=3)
            ax.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), fontsize=7.5, color="#555", va="bottom")
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), fontsize=7.5, color="#555", va="bottom", ha="right")
        else:
            ax.text(0.5, 0.5, "log not found", ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("optimizer step", fontsize=8)
        _despine(ax)
    axes[5].axis("off")
    axes[5].text(
        0.5,
        0.5,
        "Note: forward-KL token-level\nU-turns upward late in training —\n"
        "the 0.5B student memorises 500\ndistributions, then drifts.\n"
        "(See COMPARISON.md.)",
        ha="center",
        va="center",
        fontsize=9,
        color="#444",
    )
    fig.suptitle("Real logged loss per method (15-epoch runs)", fontsize=11)
    fig.tight_layout()
    return fig
