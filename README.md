# FIFA Match Predictor

Three independent models that predict the **Win / Draw / Loss** outcome of a
match between any two international (FIFA) teams, trained on the **last 10 years**
of international results.

| Model | File | Idea |
|-------|------|------|
| **Logic / dominance graph** | `graph_model.py` | A directed graph where *every arrow points to the team that won*. Dominance is transitive: if `A → B` (B beat A) and `B → C` (C beat B) then **C beats A**. Predictions come with a justification chain `t1 → t2 → … → winner`. |
| **Elo** | `elo_model.py` | World-Football-Elo-style ratings (tournament-weighted K, goal-difference multiplier, home advantage). A draw model learned from history turns the rating gap into calibrated W/D/L probabilities. |
| **Neural net** | `nn_model.py` | An **entity-embedding network** (PyTorch): a learned vector per team + context features (venue, match importance, recent form) → MLP → softmax W/D/L. Trained by `train_nn.py`; beats Elo on a temporal holdout. |

The desktop app (`predict.py`) runs **all three** models for any matchup and
draws the graph's reasoning live.

---

## Data

Source: [`martj42/international_results`](https://github.com/martj42/international_results)
— every international match since 1872, updated continuously.

`data.py` downloads and caches it to `data/results.csv`, then exposes the cleaned
**played** matches from the last 10 years. Non-FIFA sides (CONIFA teams, regional
and dependency teams such as Yorkshire, Northern Cyprus, Jersey, Greenland, Åland…)
are filtered out by default, plus the handful of non-FIFA CONCACAF associates that
play in the Gold Cup (French Guiana, Guadeloupe, Martinique, Saint Martin, Sint
Maarten, Bonaire). What's left is matches between the **211 FIFA member
associations** (~9,000 matches over the window).

```bash
python data.py          # download (if needed) + print a summary
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # if you don't have one
pip install -r requirements.txt
```

`tkinter` is part of the Python standard library. On a stripped-down Linux you
may need the OS package (`sudo apt install python3-tk`).

---

## Usage

### 1. The predictor window (both models)

```bash
python predict.py
```

Pick two teams, optionally toggle **Neutral venue**, and hit **Predict**. You get:

* **Elo** — calibrated Win / Draw / Loss bar, ratings and ranks.
* **Logic graph** — the predicted winner, the transitive justification chain, the
  head-to-head record, and a **live drawing** of the reasoning path (arrows point
  to the winner of each matchup; the green node is the pick).

### 2. Elo rankings + a specific prediction

```bash
python elo_rankings.py                       # top-25 ranking, saves elo_rankings.csv
python elo_rankings.py --all                 # every ranked team
python elo_rankings.py --predict Brazil France
python elo_rankings.py --predict Brazil France --home   # Brazil at home
```

The full ranking is always written to `elo_rankings.csv`.

### 3. Train the neural network + report metrics

```bash
python train_nn.py
```

Trains the entity-embedding net on a **temporal holdout** (older matches → train,
most recent → test, so there is no look-ahead leakage), prints the metrics table
(below), saves the model to `nn_model.pt` and metrics to `nn_metrics.json`. Once
trained, `predict.py` automatically shows a third "Neural net" panel.

### 4. The interactive graph of every team (browser)

```bash
python graph_visualize.py                        # opens an interactive HTML graph of ALL teams
python graph_visualize.py --no-open --save fifa_graph.html
python graph_visualize.py --path Tonga Andorra   # matplotlib view of a single reasoning path
```

The default builds a standalone `fifa_graph.html` (pyvis / vis.js): **all 211
teams**, coloured by confederation, sized by Elo, edges pointing loser→winner.
Drag to pan, scroll to zoom, hover a team to highlight its matches, and use the
dropdown to jump to any team.

---

## How the graph model decides

For a query **A vs B** it searches for a directed path between them:

* a path `A → … → B` means every arrow leads toward **B**, so **B wins**;
* a path `B → … → A` means **A wins**.

It prefers the **fewest hops** (the most direct argument) and, among equally short
paths, the one built from the strongest links. A direct meeting is a 1-hop path
and always wins (`confidence: high`). When the two teams have never met, the chain
might be several hops long (`confidence: low`). Edges are not single games: every
meeting in the 10-year window is aggregated into one **net dominance** score,
weighted by recency (3-year half-life) and margin of victory, and the arrow is
drawn toward whoever comes out on top. Dead-even pairs get no edge.

## How the Elo model decides

Ratings start at 1500 and update after every match by `K · G · (S − E)` where `E`
is the logistic expectation, `K` weights the match (World Cup ≫ friendly) and `G`
scales with the goal margin. The expected score mixes wins and draws, so a draw
model `P(draw) = a·exp(−(Δ/s)²)` is fit to the historical draw rate; win/loss
probabilities then follow from `E = P(win) + ½·P(draw)`, keeping everything
consistent with the rating gap.

## How the neural network decides

An **entity-embedding network** — the most suitable architecture when the inputs
are categorical entities (the two teams) plus a little context:

```
home team ─► Embedding(team) ─┐
away team ─► Embedding(team) ─┼─ concat ─► MLP (ReLU + dropout) ─► softmax → P(H), P(D), P(A)
context ─────────────────────┘
```

* A shared **team embedding** (16-d) learns a latent vector per team. Unlike a
  single Elo scalar a vector can capture *non-transitive* matchup effects once
  the MLP mixes the two teams' vectors.
* **Context features**, all known before kick-off (no leakage): neutral venue,
  match importance, and each team's recent form (points- and goal-difference-per-
  game over its last 10 matches).
* Trained with cross-entropy (= the log-loss we report), so probabilities stay
  calibrated. Regularised with dropout + weight decay + early stopping (~10k
  params, ~9k matches).

### Metrics (held-out temporal test set, ~1,350 most-recent matches)

Evaluated on matches the model never saw (most recent 15% by date, Dec 2024 →
Jun 2026). The Elo baseline is re-fit on the pre-test matches only; "prior"
predicts the training H/D/A frequencies. Lower is better for log-loss / RPS /
Brier. The **embeddings-only** column is the same network with the rolling-form
features removed — an ablation that isolates the architecture's contribution.

| metric | **NN (full)** | NN (embeddings only) | Elo | Prior |
|--------|:---:|:---:|:---:|:---:|
| accuracy | **0.610** | 0.608 | 0.585 | 0.483 |
| log loss | **0.872** | 0.874 | 0.880 | 1.048 |
| RPS | **0.171** | 0.171 | 0.174 | 0.227 |
| Brier | **0.513** | 0.513 | 0.520 | 0.631 |
| macro F1 | **0.464** | 0.450 | 0.429 | 0.217 |

The net beats Elo on every metric. Two honest caveats, both checked:

* **It's the architecture, not just hand-built features.** The *embeddings-only*
  network already beats Elo (log-loss 0.874 vs 0.880); the form features add
  little on average. So the gain comes from the learned team embeddings, not from
  feeding the net extra inputs Elo doesn't see.
* **It's robust, and the margin is small.** Over 5 random seeds the full net
  averages log-loss **0.873 ± 0.004** (RPS 0.171 ± 0.001) and beats Elo's 0.880
  in **4 / 5** seeds. The win is consistent but modest — football is high-variance
  and Elo is already strong.

**Note on draws:** a draw is almost never the single most-likely outcome, so
*argmax* accuracy/F1 for draws is low for every model (the net predicts a few;
Elo predicts none). The probabilistic scores — log-loss, RPS, Brier — are the
proper measures, and there the net assigns sensibly more mass to draws and scores
best.

## Caveats

* **Football isn't transitive.** "A beat B, B beat C, therefore A beats C" is a
  *logical* argument, not a law — upsets happen. The graph model embraces this on
  purpose (it's what was asked for); the Elo model is the calibrated, probabilistic
  counterpart. Seeing the two **agree** is a stronger signal than either alone.
* The graph's transitive *reach* is not a power ranking (the graph is one big
  cycle-rich component). Use the Elo table for ranking; the graph for reasoning.

## Files

```
data.py             fetch + cache + clean the dataset (last 10y, FIFA-only) + confederations
elo_model.py        Elo engine + learned draw model + W/D/L prediction
graph_model.py      dominance graph + transitive pathfinding + justification
nn_model.py         entity-embedding neural net (PyTorch) + causal features + predictor
train_nn.py         train the net on a temporal split, report metrics vs Elo/prior
metrics.py          accuracy / log-loss / RPS / Brier / per-class F1
graph_visualize.py  interactive browser graph (pyvis) + matplotlib reasoning path
elo_rankings.py     CLI: print/save Elo ranking + a specific prediction
predict.py          tkinter GUI: all three models + live graph
requirements.txt
data/results.csv    cached dataset
elo_rankings.csv    generated ranking
nn_model.pt         trained network (generated by train_nn.py)
nn_metrics.json     held-out metrics (generated by train_nn.py)
fifa_graph.html     interactive all-teams graph (generated by graph_visualize.py)
```

***Note: most of this readme was generated by claude because I was too lazy to do it, but it seems to be mostly a good readme and the overall claims it makes appear to be accurate. If any of it is wrong let me know and I will fix it.***

*Disclaimer: Claude Code was used in the development of this repo, particularly in building the network embedding architecture and visualizing the logic graphs.*
