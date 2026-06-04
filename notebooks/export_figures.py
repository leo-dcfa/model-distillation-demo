"""Render the notebook figures to static PNGs for the README.

    uv run python notebooks/export_figures.py

Writes to ``assets/`` at the repo root. The interactive versions (with sliders)
live in ``notebooks/distillation_viz.py``; these are the default-parameter
snapshots GitHub can render inline.
"""

from pathlib import Path

import matplotlib

import figures as F


matplotlib.use("Agg")

ASSETS = Path(__file__).resolve().parent.parent / "assets"

FIGURES = {
    "method1_sequence_level.png": lambda: F.sequence_level(0.55),
    "method2_token_match.png": lambda: F.token_level_match(2.0),
    "method2_kl_forward.png": lambda: F.kl_direction(reverse=False),
    "method2_kl_reverse.png": lambda: F.kl_direction(reverse=True),
    "method3_on_policy.png": lambda: F.on_policy(0.5),
    "method4_uld_alignment.png": lambda: F.uld_alignment(),
    "method4_uld_sorted_topk.png": lambda: F.uld_sorted_topk(6),
    "loss_curves.png": lambda: F.loss_curves(),
}


def main():
    ASSETS.mkdir(exist_ok=True)
    for name, build in FIGURES.items():
        fig = build()
        out = ASSETS / name
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"wrote {out.relative_to(ASSETS.parent)}")


if __name__ == "__main__":
    main()
