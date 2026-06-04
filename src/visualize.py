"""Look inside the distilled students with TransformerLens and matplotlib.

Produces four figures under ``figures/``:

  1. logit_lens.png      Per-layer probability of the correct answer token for the
                         base student vs each distilled student. Reading the
                         residual stream layer-by-layer through the unembedding
                         ("logit lens") shows *where in the network* each
                         distillation method builds the answer.

  2. attention_grid.png  Every attention head at one layer of one student, on a
                         short word problem — a literal look inside the attention.

  3. attention_diff.png  Base vs distilled attention for the single head that
                         distillation changed the most, plus the difference. The
                         LoRA adapters only edit attention, so this is *the* change
                         distillation made, isolated.

  4. token_shift.png     The tokens distillation most boosts / suppresses at the
                         answer position, relative to the base student.

Run:  uv run python -m src.visualize
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.constants import SYSTEM_PROMPT  # noqa: E402
from src.interp import MODELS, free, load_hooked  # noqa: E402


FIG_DIR = Path("figures")

# Arithmetic completions the base student already gets right, so the logit-lens
# curves are comparable across models. Each value is the (single-token) answer
# digit we track through the layers.
LOGIT_LENS_PROMPTS: dict[str, str] = {
    "2+2=": "4",
    "9*9=": "8",      # 81
    "12-5=": "7",
    "Q: 7+5\nA: 7+5=": "1",  # 12
}

# A short word problem used for the attention figures (bare completion keeps the
# token axes readable).
WORD_PROBLEM = "Sarah had 5 apples. She bought 3 more. Now Sarah has"

# The same problem posed through the chat template the students were trained with,
# used for the token-shift figure so the models are in their trained regime.
CHAT_QUESTION = "Sarah had 5 apples. She bought 3 more. How many apples does Sarah have now?"

# Which distilled student to feature in the single-model attention figures.
FEATURED = "sequence_level"
FEATURED_ADAPTER = "distilled_sequence_level"
ATTN_LAYER = 12  # a middle layer, of 24


def logit_lens_curve(model, prompt: str, answer: str) -> np.ndarray:
    """Probability assigned to ``answer`` at each accumulated-residual point.

    Returns an array of length ``n_layers + 1`` (embeddings, then after each
    block), each entry the probability the logit lens puts on the answer token
    at the final position.
    """
    target = model.to_single_token(answer)
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        # [n_points, batch, pos, d_model], with the final LayerNorm applied.
        accum = cache.accumulated_resid(layer=-1, incl_mid=False, apply_ln=True)
        if accum.ndim == 4:
            accum = accum[:, 0]  # drop the batch dim -> [n_points, pos, d_model]
        logits = accum[:, -1, :] @ model.W_U  # [n_points, vocab]
        probs = logits.softmax(dim=-1)[:, target]
    return probs.float().cpu().numpy()


def collect(args) -> dict:
    """Single pass over the models, gathering everything the figures need."""
    data: dict = {"logit_lens": {}, "answer_logits": {}}

    for label, adapter in MODELS:
        print(f"loading {label} ...")
        model = load_hooked(adapter)

        # 1. Logit-lens curves, averaged over the arithmetic prompts.
        curves = [logit_lens_curve(model, p, a) for p, a in LOGIT_LENS_PROMPTS.items()]
        data["logit_lens"][label] = np.mean(curves, axis=0)

        # 4. Final-position probabilities on the chat-templated problem, where the
        #    distilled students are in their trained regime (for the token-shift figure).
        chat_prompt = model.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": CHAT_QUESTION}],
            tokenize=False,
            add_generation_prompt=True,
        )
        with torch.no_grad():
            chat_logits = model(chat_prompt)
        data["answer_logits"][label] = chat_logits[0, -1].softmax(-1).float().cpu().numpy()

        # 2/3. Cache attention patterns for the base and the featured student.
        if label in ("base", FEATURED):
            with torch.no_grad():
                _, cache = model.run_with_cache(WORD_PROBLEM)
            patterns = torch.stack([cache["pattern", layer][0] for layer in range(model.cfg.n_layers)])
            data.setdefault("attn", {})[label] = patterns.float().cpu().numpy()  # [layer, head, q, k]
            data["str_tokens"] = model.to_str_tokens(WORD_PROBLEM)
            data["n_heads"] = model.cfg.n_heads
            data["n_layers"] = model.cfg.n_layers

        if label == "base":
            data["to_str"] = model.to_str_tokens  # for labelling the token-shift bars
            data["vocab_str"] = model.to_string

        free(model)

    return data


def plot_logit_lens(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, _ in MODELS:
        curve = data["logit_lens"][label]
        xs = np.arange(len(curve))
        ax.plot(xs, curve, marker="o", markersize=3, linewidth=2 if label == "base" else 1.6, label=label)
    ax.set_xlabel("residual stream depth  (0 = embeddings, 24 = final layer)")
    ax.set_ylabel("P(correct answer token)  via logit lens")
    ax.set_title("Where each distillation method builds the answer\n(mean over simple arithmetic prompts)")
    ax.grid(alpha=0.3)
    ax.legend(title="student")
    fig.tight_layout()
    out = FIG_DIR / "logit_lens.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_attention_grid(data: dict) -> None:
    patterns = data["attn"][FEATURED][ATTN_LAYER]  # [head, q, k]
    tokens = data["str_tokens"]
    n_heads = data["n_heads"]
    ncols = 5
    nrows = int(np.ceil(n_heads / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.3 * nrows))
    axes = np.array(axes).reshape(-1)
    for head in range(n_heads):
        ax = axes[head]
        ax.imshow(patterns[head], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"head {head}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for extra in range(n_heads, len(axes)):
        axes[extra].axis("off")
    fig.suptitle(f"{FEATURED} student — all {n_heads} attention heads at layer {ATTN_LAYER}\nprompt: {WORD_PROBLEM!r}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "attention_grid.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_attention_diff(data: dict) -> None:
    base = data["attn"]["base"]  # [layer, head, q, k]
    dist = data["attn"][FEATURED]
    tokens = data["str_tokens"]

    # Pick the (layer, head) whose attention distillation changed the most.
    delta = np.abs(dist - base).reshape(base.shape[0], base.shape[1], -1).sum(-1)
    layer, head = np.unravel_index(int(delta.argmax()), delta.shape)
    b, d = base[layer, head], dist[layer, head]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mat, title, cmap, kw in [
        (axes[0], b, "base", "viridis", {"vmin": 0, "vmax": 1}),
        (axes[1], d, FEATURED, "viridis", {"vmin": 0, "vmax": 1}),
        (axes[2], d - b, f"{FEATURED} − base", "RdBu_r", {"vmin": -0.5, "vmax": 0.5}),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", **kw)
        ax.set_title(title)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_xlabel("key (attended to)")
        if ax is axes[0]:
            ax.set_ylabel("query (attending from)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"What distillation changed — layer {layer}, head {head} (the most-changed head)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "attention_diff.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_token_shift(data: dict, top_k: int = 12) -> None:
    base = data["answer_logits"]["base"]  # probabilities
    fig, axes = plt.subplots(1, len(MODELS) - 1, figsize=(4.2 * (len(MODELS) - 1), 5.5))
    axes = np.array(axes).reshape(-1)
    distilled = [(lbl, ad) for lbl, ad in MODELS if lbl != "base"]
    for ax, (label, _) in zip(axes, distilled, strict=True):
        diff = data["answer_logits"][label] - base  # Δ probability
        order = np.argsort(diff)
        picks = np.concatenate([order[-top_k:][::-1], order[:top_k][::-1]])
        vals = diff[picks] * 100  # percentage points
        names = [repr(data["vocab_str"](int(t))) for t in picks]
        colors = ["#2a7" if v > 0 else "#c44" for v in vals]
        ax.barh(range(len(picks)), vals, color=colors)
        ax.set_yticks(range(len(picks)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Δ P(token)  (percentage points)")
    fig.suptitle(
        "What the first answer token becomes after distillation\n"
        f"boosted (green) / suppressed (red) vs base, chat-templated prompt: {CHAT_QUESTION!r}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "token_shift.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures", help="Directory to write figures to.")
    args = parser.parse_args()

    global FIG_DIR
    FIG_DIR = Path(args.out)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    data = collect(args)

    print("plotting ...")
    plot_logit_lens(data)
    plot_attention_grid(data)
    plot_attention_diff(data)
    plot_token_shift(data)
    print(f"done — see {FIG_DIR}/")


if __name__ == "__main__":
    main()
