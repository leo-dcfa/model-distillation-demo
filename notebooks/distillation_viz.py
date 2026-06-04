# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "matplotlib", "numpy"]
# ///
"""Interactive visualisations of the four distillation methods in this repo.

Run it:

    uv run marimo edit notebooks/distillation_viz.py     # interactive
    uv run marimo run  notebooks/distillation_viz.py     # read-only app

Each section mirrors one method in the README and is driven by a small control
(slider / toggle) so you can *see* the mechanism — what signal the student gets,
and what gets thrown away. All the plotting lives in ``notebooks/figures.py``;
the cells below are just the wiring from a UI control to a figure.
"""

import marimo


__generated_with = "0.23.8"
app = marimo.App(
    width="medium",
    layout_file="layouts/distillation_viz.slides.json",
)


@app.cell
def _():
    import sys
    from pathlib import Path

    # Make `figures.py` importable whether marimo is launched from the repo root
    # or from inside notebooks/.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo

    import figures as F

    return F, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Four flavours of distillation, visualised

    The README describes four ways to pour a big **teacher** model into a small
    **student**. Words only get you so far — below, each method is an interactive
    picture. The recurring question across all four:

    > *At each token, what does the teacher know, and how much of it does the
    > student actually get to see?*

    <span style="color:#2f6db5">**blue = teacher / target**</span> &nbsp;·&nbsp;
    <span style="color:#d1772e">**orange = student**</span> &nbsp;·&nbsp;
    <span style="color:#c2403d">**red = signal the method keeps**</span> &nbsp;·&nbsp;
    <span style="color:#c9ccd1">**grey = signal thrown away**</span>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The payoff first · what each method actually buys you

    Before the mechanisms, the scoreboard from
    [`COMPARISON.md`](../COMPARISON.md): a `Qwen2.5-0.5B` student evaluated on 100
    held-out GSM8K problems after 15 epochs of distillation from a 3B teacher
    (`SmolLM2-1.7B` for ULD). The un-tuned student scores **9%**; distillation
    lifts it to as high as **44%**.
    """)
    return


@app.cell
def _(F):
    F.headline_results()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The ordering — on-policy ≈ sequence-level > token-level >> cross-tokenizer — is
    the robust signal; the 1-point on-policy/sequence-level gap is within noise at
    n=100. Two caveats worth carrying into the rest of the notebook:

    - **Token-level KL (37%) is *underperforming* here**, not broken. Its loss
      U-turns mid-run — the 0.5B student memorises 500 teacher distributions by
      ~epoch 5, then drifts. Fewer epochs or early stopping would likely put it
      above sequence-level (see the loss panel at the end).
    - **On-policy's tie comes at ~20–40× the compute** (114 min vs ~3–6 min). It
      should pull ahead on long-horizon tasks where the student's own mistakes
      compound — GSM8K's short answers aren't that regime.

    Now, the mechanisms that produce these numbers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Method 1 · Sequence-level (SFT on teacher text)

    The teacher writes a full solution; the student is trained with plain
    next-token cross-entropy to reproduce it. At every position the target is a
    single token — the teacher's **argmax** — collapsed to a one-hot. Everything
    the teacher *almost* said is discarded. Drag the slider to see how much of the
    teacher's distribution survives the collapse to a one-hot label.
    """)
    return


@app.cell
def _(mo):
    m1_conf = mo.ui.slider(
        start=0.35,
        stop=0.95,
        step=0.05,
        value=0.55,
        label="Teacher confidence in its top token",
        show_value=True,
    )
    m1_conf
    return (m1_conf,)


@app.cell
def _(F, m1_conf):
    F.sequence_level(m1_conf.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaway.** Cheap and API-only (you just need `generate`), tokenizer-
    agnostic — but the grey bars are real information the student never sees. If the
    teacher was 55% on `"12"` and 20% on `"twelve"`, sequence-level tells the
    student `"twelve"` was simply *wrong*.

    ---
    ## Method 2 · Token-level KL (forward vs reverse)

    Now keep the whole distribution. At every position, match the student's full
    next-token distribution to the teacher's by minimising a KL divergence. Two
    knobs change the character of the result:

    - **Temperature `T`** softens both distributions before comparing — exposing the
      teacher's ranking of the *unlikely* tokens (the "dark knowledge").
    - **KL direction**: *forward* `KL(teacher‖student)` is **mode-covering** (hedge,
      spread mass to cover every teacher behaviour); *reverse* `KL(student‖teacher)`
      is **mode-seeking** (commit to one mode, stay sharp).
    """)
    return


@app.cell
def _(mo):
    m2_temp = mo.ui.slider(start=1.0, stop=4.0, step=0.5, value=2.0, label="Temperature T", show_value=True)
    m2_dir = mo.ui.radio(
        options=["forward — mode-covering", "reverse — mode-seeking"],
        value="forward — mode-covering",
        label="KL direction",
    )
    mo.hstack([m2_temp, m2_dir], justify="start", gap=3)
    return m2_dir, m2_temp


@app.cell
def _(F, m2_temp):
    F.token_level_match(m2_temp.value)
    return


@app.cell
def _(F, m2_dir):
    F.kl_direction(reverse=m2_dir.value.startswith("reverse"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaway.** Token-level gives the student a *dense* signal — the full shape at
    every position — but needs white-box access and a **matching tokenizer**.
    Forward KL produces a hedging, hallucination-prone student; reverse KL (used by
    on-policy below, following MiniLLM/GKD) produces a focused one.

    ---
    ## Method 3 · On-policy distillation (GKD)

    Methods 1–2 train on prefixes the *teacher* wrote, so the student never
    practises recovering from **its own** mistakes — that's exposure bias. On-policy
    flips the data source: the **student generates**, the teacher **scores** that
    rollout's distribution, and a fraction `α` of steps use these fresh on-policy
    rollouts (the rest reuse teacher text for stability). Slide the mix and watch the
    train/test gap close.
    """)
    return


@app.cell
def _(mo):
    m3_ratio = mo.ui.slider(start=0.0, stop=1.0, step=0.1, value=0.5, label="On-policy fraction α", show_value=True)
    m3_ratio
    return (m3_ratio,)


@app.cell
def _(F, m3_ratio):
    F.on_policy(m3_ratio.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaway.** As `α → 1` the orange training distribution slides onto the dashed
    test distribution — the student practises on exactly the (messier) prefixes it
    will face at inference. The cost is brutal: every on-policy step runs a full
    sampling pass through the student first, which is why on-policy is ~20–40× slower
    here (114 min vs ~3–6 min for the off-policy methods).

    ---
    ## Method 4 · Cross-tokenizer (ULD)

    Token-level KL is undefined when teacher and student have **different
    vocabularies** — their distributions live in non-comparable spaces. ULD bridges
    that with two tricks, both visible below:

    1. **Align by character offset.** Each student response token is matched to the
       teacher token whose span *ends at the closest character*.
    2. **Sorted top-K matching.** At each aligned pair, take the top-K probabilities
       from each side, **sort** them, and compare those shapes — token identity is
       thrown away, so the comparison lives in a vocab-independent K-dim space.
    """)
    return


@app.cell
def _(F):
    F.uld_alignment()
    return


@app.cell
def _(mo):
    m4_topk = mo.ui.slider(start=3, stop=12, step=1, value=6, label="top-K", show_value=True)
    m4_topk
    return (m4_topk,)


@app.cell
def _(F, m4_topk):
    F.uld_sorted_topk(m4_topk.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Takeaway.** ULD is the only method here that crosses model families — but it's
    lossy on three axes at once (approximate alignment, identity-blind sorted
    matching, truncated tail), which is why it trails the same-tokenizer methods by
    25+ points in `COMPARISON.md`.

    ---
    ## Epilogue · the real training curves

    Everything above is a cartoon of the *mechanism*. Here are the actual loss
    trajectories logged by `src/logger.py` during the 15-epoch runs. Magnitudes
    aren't comparable across panels (different losses, temperatures, scales) — only
    the *shape within a panel* is meaningful.
    """)
    return


@app.cell
def _(F):
    F.loss_curves()
    return


if __name__ == "__main__":
    app.run()
