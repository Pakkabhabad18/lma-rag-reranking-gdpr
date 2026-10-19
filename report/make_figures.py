# Figures for the report, drawn from the results/ files saved by rag_assignment.ipynb.
# Run from inside the report/ folder:  python make_figures.py
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.environ.get("RES", "../results")
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

GREY, LIGHT, BLUE, RED, INK, QUIET = "#9a9993", "#e9e8e4", "#2a78d6", "#d03b3b", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#b8b7b1", "axes.linewidth": 0.7,
    "axes.labelcolor": QUIET, "xtick.color": QUIET, "ytick.color": QUIET, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

sweep = pd.read_csv(f"{RES}/failure_sweep.csv")
summary = pd.read_csv(f"{RES}/retrieval_summary.csv", index_col=0)
lat = pd.read_csv(f"{RES}/latency_retrieval.csv", index_col=0)
e2e = pd.read_csv(f"{RES}/latency_end_to_end.csv", index_col=0)
full = pd.read_csv(f"{RES}/full_cross_encoder_ranks.csv")

# ---------- rank shift of the gold chunk (full width) ----------
order = ["E1", "N11", "N13", "K1", "K3"]
cases = sweep.set_index("qid").loc[order].reset_index()
short = {"E1": "E1  always inform people of a breach?", "N11": "N11  no approval needed (adequate country)?",
         "N13": "N13  can the DPO be fired?", "K1": "K1  delete everything about me?", "K3": "K3  maximum penalty?"}
fig, ax = plt.subplots(figsize=(6.4, 2.1))
ax.axvspan(0.5, 3.5, color=LIGHT, zorder=0)
ax.text(2.0, -0.8, "top 3 = sent to Gemini", ha="center", va="center", color=QUIET, fontsize=7.5)
for i, r in cases.iterrows():
    y = len(cases) - 1 - i
    b, c = int(r.gold_bi_rank), int(r.gold_ce_rank)
    col = BLUE if r.outcome.startswith("fixed") else RED
    ax.annotate("", xy=(c, y), xytext=(b, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=4, shrinkB=4))
    ax.plot(b, y, "o", color=GREY, ms=6, zorder=3)
    ax.plot(c, y, "o", color=col, ms=6, zorder=3)
    ax.text(b + (0.3 if b > c else -0.3), y + 0.25, f"#{b}", ha="center", fontsize=7.5, color=QUIET)
    ax.text(c + (-0.3 if b > c else 0.3), y + 0.25, f"#{c}", ha="center", fontsize=7.5, color=col)
ax.set_yticks(range(len(cases)))
ax.set_yticklabels([short[q] for q in reversed(order)], color=INK)
ax.set_xlim(0.5, 10.5); ax.set_ylim(-1.2, len(cases) - 0.4)
ax.set_xticks(range(1, 11)); ax.set_xlabel("rank of the correct chunk (1 = top)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.plot([], [], "o", color=GREY, label="bi-encoder (Pipeline 1)")
ax.plot([], [], "o", color=BLUE, label="after re-ranking (Pipeline 2)")
ax.plot([], [], "o", color=RED, label="after re-ranking, moved down")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False, fontsize=7.5)
fig.savefig(f"{OUT}/rank_shift.pdf"); plt.close(fig)

# ---------- Hit@1 / Hit@3 / MRR@10 (half width) ----------
metrics = ["Hit@1", "Hit@3", "MRR@10"]
p1 = summary.loc[metrics, "Pipeline 1 (bi-encoder)"].values
p2 = summary.loc[metrics, "Pipeline 2 (cross-encoder)"].values
fig, ax = plt.subplots(figsize=(3.2, 2.0))
x = np.arange(len(metrics)); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, p1, w, color=GREY, label="Pipeline 1")
b2 = ax.bar(x + w / 2 + 0.01, p2, w, color=BLUE, label="Pipeline 2 (re-ranked)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.3f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(metrics, color=INK)
ax.set_ylim(0, 0.68); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
fig.savefig(f"{OUT}/metrics.pdf"); plt.close(fig)

# ---------- retrieval latency per step (half width) ----------
steps = ["query embedding", "FAISS search (434 chunks)", "cross-encoder (10 pairs)"]
labels = ["query embedding", "FAISS search", "cross-encoder\n(10 pairs)"]
med = lat.loc[steps, "median (ms)"].values
p95 = lat.loc[steps, "p95 (ms)"].values
fig, ax = plt.subplots(figsize=(3.2, 1.55))
y = np.arange(len(steps))[::-1]
ax.barh(y, med, height=0.55, color=[GREY, GREY, BLUE])
ax.errorbar(med, y, xerr=[np.zeros_like(med), p95 - med], fmt="none", ecolor=QUIET, elinewidth=0.9, capsize=2.5)
for yi, m, p in zip(y, med, p95):
    ax.text(p + 4, yi, f"{m:.2f} ms" if m < 1 else f"{m:.1f} ms", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels, color=INK, fontsize=7.5)
ax.set_xlabel("ms per question (bar = median, whisker = p95)")
ax.set_xlim(0, 260); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.savefig(f"{OUT}/latency.pdf"); plt.close(fig)

# ---------- scaling (half width, log-log) ----------
per_pair_ms = 4712 / 434                 # cross-encoder over all 434 chunks, measured
embed_ms = lat.loc["query embedding", "median (ms)"]
search434 = lat.loc["FAISS search (434 chunks)", "median (ms)"]
per_vec = 3.3 / 100_000                  # flat search over 100,000 vectors, measured
rerank_ms = lat.loc["cross-encoder (10 pairs)", "median (ms)"]
llm_ms = e2e["LLM median (ms)"].median()
n = np.logspace(np.log10(434), 6, 60)
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.plot(n, per_pair_ms * n / 1000, color=RED, lw=1.8, label="cross-encoder on every chunk")
ax.plot(n, (embed_ms + np.maximum(search434, per_vec * n) + rerank_ms) / 1000, color=BLUE, lw=1.8, label="bi-encoder top 10 + re-rank")
ax.axhline(llm_ms / 1000, color=QUIET, lw=0.9, ls="--")
ax.text(2.2e5, llm_ms / 1000 * 1.35, "one Gemini call", fontsize=7, color=QUIET, ha="center")
ax.plot(434, per_pair_ms * 434 / 1000, "o", color=RED, ms=4.5)
ax.plot(434, (embed_ms + search434 + rerank_ms) / 1000, "o", color=BLUE, ms=4.5)
ax.text(1e6, per_pair_ms * 1e6 / 1000 * 0.45, "3.0 h", fontsize=7, color=RED, ha="right", va="top")
ax.text(1e6, 0.35, "0.2 s", fontsize=7, color=BLUE, ha="right")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("chunks in the collection"); ax.set_ylabel("seconds per question")
ax.set_xlim(300, 1.1e6); ax.set_ylim(0.05, 3e4)
ax.grid(True, which="major", color="#ecebe7", lw=0.5)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.savefig(f"{OUT}/scaling.pdf"); plt.close(fig)

# ---------- missed gold chunks ranked over all 434 chunks (half width) ----------
full = full.sort_values("gold rank, cross-encoder over all chunks")
fig, ax = plt.subplots(figsize=(3.2, 2.5))
ax.axvspan(0.8, 10.5, color=LIGHT, zorder=0)
ax.text(2.9, -0.95, "top 10 (what the re-ranker sees)", ha="center", fontsize=6.5, color=QUIET)
for i, r in enumerate(full.itertuples()):
    b, c = r[3], r[4]
    y = len(full) - 1 - i
    ax.plot([b, c], [y, y], color="#cfcec9", lw=1.2, zorder=1)
    ax.plot(b, y, "o", color=GREY, ms=4.5, zorder=3)
    ax.plot(c, y, "o", color=BLUE, ms=4.5, zorder=3)
ax.set_yticks(range(len(full))); ax.set_yticklabels(list(full.qid)[::-1], color=INK)
ax.set_xscale("log"); ax.set_xlim(0.8, 200); ax.set_ylim(-1.4, len(full) - 0.4)
ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["1", "3", "10", "30", "100"])
ax.set_xlabel("rank among all 434 chunks (log)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.plot([], [], "o", color=GREY, label="bi-encoder"); ax.plot([], [], "o", color=BLUE, label="cross-encoder")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
fig.savefig(f"{OUT}/full_ranks.pdf"); plt.close(fig)

print("figures written to", OUT)
