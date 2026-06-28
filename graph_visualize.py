"""
graph_visualize.py
===================
Render the dominance graph from :mod:`graph_model`.

Two views:

* ``draw_prediction(gm, prediction, ax)`` — the "live" view used by the GUI:
  the reasoning path (loser ──► … ──► winner) laid out left-to-right with the
  immediate neighbourhood of each endpoint for context. Path edges are bold and
  coloured; arrows point to the winner of every matchup.

* ``draw_overview(gm, ax)`` — a legible snapshot of the strongest teams and the
  dominance edges among them.

The drawing functions only ever touch the matplotlib ``Axes`` handed to them,
so they work both with pyplot (CLI) and with an embedded ``FigureCanvasTkAgg``
(the predict.py GUI) without fighting over the backend.
"""
from __future__ import annotations

import html
import os
import webbrowser

import networkx as nx

# Confederation colours for the interactive browser graph
CONF_COLORS = {
    "UEFA": "#3498db",        # blue
    "CONMEBOL": "#f1c40f",    # yellow
    "CAF": "#2ecc71",         # green
    "AFC": "#e74c3c",         # red
    "CONCACAF": "#e67e22",    # orange
    "OFC": "#9b59b6",         # purple
    "Other": "#95a5a6",       # grey
}

# Colour palette
C_PATH_EDGE = "#c0392b"     # bold red for the reasoning path
C_PATH_NODE = "#f5b041"     # amber for nodes on the path
C_WINNER = "#27ae60"        # green for the predicted winner
C_LOSER = "#aab7b8"         # grey for the predicted loser (start of path)
C_CONTEXT_NODE = "#d6eaf8"  # pale blue for context teams
C_CONTEXT_EDGE = "#cccccc"  # light grey for context edges


def _context_neighbours(gm, team: str, limit: int) -> list[str]:
    """A few strongest neighbours of ``team`` (either direction) for context."""
    g = gm.graph
    if team not in g:
        return []
    nbrs = []
    for u, v, d in g.in_edges(team, data=True):     # teams team beats
        nbrs.append((u, d.get("strength", 0.0)))
    for u, v, d in g.out_edges(team, data=True):    # teams that beat team
        nbrs.append((v, d.get("strength", 0.0)))
    nbrs.sort(key=lambda kv: kv[1], reverse=True)
    seen, out = set(), []
    for name, _ in nbrs:
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= limit:
            break
    return out


_SPINE_DX = 2.2   # horizontal gap between consecutive nodes on the spine


def _stack_y(count: int) -> list[float]:
    """Vertical offsets centred on 0, e.g. count=3 -> [-1.4, 0, 1.4]."""
    if count <= 0:
        return []
    if count == 1:
        return [1.6]
    step = 1.4
    start = -step * (count - 1) / 2.0
    return [start + i * step for i in range(count)]


def _layout(spine: list[str], ctx_left: list[str],
            ctx_right: list[str]) -> dict:
    """Path nodes on a horizontal spine; context stacked off each end.

    ``spine`` is the reasoning path (loser … winner) — or just the two queried
    teams when there is no path. ``ctx_left`` hangs off ``spine[0]`` to the
    left, ``ctx_right`` off ``spine[-1]`` to the right, so neither crowds the
    spine.
    """
    pos = {}
    for i, node in enumerate(spine):
        pos[node] = (i * _SPINE_DX, 0.0)
    left_x = -_SPINE_DX * 0.85
    right_x = (len(spine) - 1) * _SPINE_DX + _SPINE_DX * 0.85
    for node, y in zip(ctx_left, _stack_y(len(ctx_left))):
        pos[node] = (left_x, y)
    for node, y in zip(ctx_right, _stack_y(len(ctx_right))):
        pos[node] = (right_x, y)
    return pos


def draw_prediction(gm, prediction: dict, ax, max_context: int = 3) -> None:
    """Draw the reasoning subgraph for a single prediction into ``ax``."""
    ax.clear()
    ax.set_axis_off()

    team_a = prediction.get("team_a")
    team_b = prediction.get("team_b")
    path = prediction.get("path") or []
    winner = prediction.get("winner")

    # The spine is the reasoning path, or just the two teams if there's no path.
    spine = list(path) if len(path) >= 2 else [team_a, team_b]
    spine = [n for n in spine if n in gm.graph]
    if not spine:
        ax.text(0.5, 0.5, "No graph data for these teams",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Context hangs off the two ends of the spine (avoid duplicating spine nodes).
    spine_set = set(spine)
    ctx_left = [n for n in _context_neighbours(gm, spine[0], max_context)
                if n not in spine_set]
    used = spine_set | set(ctx_left)
    ctx_right = [n for n in _context_neighbours(gm, spine[-1], max_context)
                 if n not in used]

    nodes = spine_set | set(ctx_left) | set(ctx_right)
    H = gm.graph.subgraph(nodes).copy()
    pos = _layout(spine, ctx_left, ctx_right)

    path_edges = set(zip(path[:-1], path[1:])) if len(path) > 1 else set()
    ctx_edges = [e for e in H.edges if e not in path_edges]

    # Per-node colours and sizes (computed first so edges know how far to
    # inset their arrowheads — otherwise the head can hide under a big node).
    node_colors, node_sizes = [], []
    path_set = set(path)
    for node in H.nodes:
        if node == winner:
            node_colors.append(C_WINNER); node_sizes.append(1700)
        elif path and node == path[0]:
            node_colors.append(C_LOSER); node_sizes.append(1300)
        elif node in path_set:
            node_colors.append(C_PATH_NODE); node_sizes.append(1300)
        else:
            node_colors.append(C_CONTEXT_NODE); node_sizes.append(800)

    # Context edges first (faint), then the bold path on top. Passing the real
    # node_size array makes every arrowhead stop just outside its target node.
    nx.draw_networkx_edges(
        H, pos, edgelist=ctx_edges, ax=ax, edge_color=C_CONTEXT_EDGE,
        width=1.0, arrows=True, arrowstyle="-|>", arrowsize=10,
        node_size=node_sizes, alpha=0.5,
    )
    if path_edges:
        nx.draw_networkx_edges(
            H, pos, edgelist=list(path_edges), ax=ax, edge_color=C_PATH_EDGE,
            width=2.6, arrows=True, arrowstyle="-|>", arrowsize=20,
            node_size=node_sizes, min_source_margin=12, min_target_margin=12,
        )

    nx.draw_networkx_nodes(
        H, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors="#34495e", linewidths=1.2,
    )
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=8, font_weight="bold")

    # Title / caption
    if winner and path:
        method = prediction.get("method", "")
        ax.set_title(f"Graph reasoning  —  pick: {winner}  ({method})",
                     fontsize=11, fontweight="bold")
        reading = prediction.get("reading")
        if reading:
            ax.text(0.5, -0.04, reading, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color="#566573",
                    wrap=True)
    elif winner:
        ax.set_title(f"Graph reasoning  —  pick: {winner} (structural)",
                     fontsize=11, fontweight="bold")
    else:
        ax.set_title("Graph reasoning  —  no dominance path",
                     fontsize=11, fontweight="bold")
    ax.margins(0.18)


def draw_overview(gm, ax, top_n: int = 30) -> None:
    """Draw dominance edges among the ``top_n`` strongest teams."""
    ax.clear()
    ax.set_axis_off()
    ranked = gm.top_dominators(top_n)
    scores = dict(ranked)
    H = gm.graph.subgraph([t for t, _ in ranked]).copy()
    pos = nx.spring_layout(H, seed=7, k=1.1, iterations=200)

    nx.draw_networkx_edges(
        H, pos, ax=ax, edge_color=C_CONTEXT_EDGE, width=0.8, alpha=0.6,
        arrows=True, arrowstyle="-|>", arrowsize=8, node_size=600,
    )
    colors = [scores.get(n, 0) for n in H.nodes]
    nx.draw_networkx_nodes(
        H, pos, ax=ax, node_color=colors, cmap="YlGn", node_size=700,
        edgecolors="#34495e", linewidths=0.8,
    )
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=7)
    ax.set_title(
        f"Dominance graph — top {top_n} teams (arrows point to the winner)",
        fontsize=12, fontweight="bold",
    )
    ax.margins(0.12)


def _legend_html() -> str:
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:12px;'
        f'height:12px;border-radius:50%;background:{c};margin-right:6px"></span>'
        f'{name}</div>'
        for name, c in CONF_COLORS.items()
    )
    return (
        # top-right so it never covers the select-menu dropdown (top-left)
        '<div style="position:fixed;top:12px;right:12px;z-index:999;'
        'background:rgba(20,20,20,.82);color:#eee;padding:10px 14px;'
        'border-radius:8px;font:13px/1.4 Helvetica,Arial,sans-serif;'
        'max-width:240px">'
        '<b>FIFA dominance graph</b><br>'
        '<span style="font-size:11px;color:#bbb">arrow points to the winner · '
        'node size = Elo · drag to pan · scroll to zoom · click a team to '
        'highlight its matches · use the dropdown (top-left) to find a team'
        '</span><hr style="border-color:#444">'
        f'{rows}</div>'
    )


def build_interactive(gm, elo, confeds: dict, output_path: str = "fifa_graph.html",
                      open_browser: bool = True) -> str:
    """Write a standalone, pannable/zoomable HTML graph of ALL teams.

    Nodes = every team, coloured by confederation and sized by Elo; edges point
    loser → winner (faint, so the 200-node graph stays readable). Built with
    pyvis/vis.js: drag to pan, scroll to zoom, hover to highlight a team's
    matches, and a dropdown to jump to any team.
    """
    from pyvis.network import Network

    g = gm.graph
    pos = nx.spring_layout(g, seed=7, k=0.55, iterations=120)
    # scale unit-square layout into vis.js pixel space
    span = 2200.0
    ranks = {r.team: r.rank
             for r in elo.ranking_frame().itertuples(index=False)}

    # leave room for the select-menu header so the canvas doesn't overflow 100vh
    net = Network(height="calc(100vh - 60px)", width="100%", directed=True,
                  bgcolor="#15181c", font_color="#e8e8e8", notebook=False,
                  select_menu=True, cdn_resources="in_line")
    net.toggle_physics(False)

    max_strength = max((d["strength"] for *_, d in g.edges(data=True)),
                       default=1.0)
    for team in g.nodes:
        conf = confeds.get(team, "Other")
        rating = elo.rating(team)
        st = elo.teams.get(team)
        rank = ranks.get(team, "?")
        dom = gm.dominance_score(team)
        size = 8 + max(rating - 1300.0, 0) / 22.0     # ~8..40 px
        title = (
            f"<b>{html.escape(team)}</b><br>{conf}<br>"
            f"Elo {rating:.0f} (rank #{rank})<br>"
            f"{st.record if st else ''} in {st.games if st else 0} games<br>"
            f"net direct dominance {dom:+d}"
        )
        x, y = pos[team]
        net.add_node(team, label=team, title=title, shape="dot", size=size,
                     color=CONF_COLORS.get(conf, CONF_COLORS["Other"]),
                     x=float(x * span), y=float(-y * span), physics=False)

    for loser, winner, d in g.edges(data=True):
        w = 0.4 + 3.0 * (d["strength"] / max_strength)
        net.add_edge(loser, winner, width=w,
                     color="rgba(180,180,180,0.18)",
                     title=f"{html.escape(winner)} dominates "
                           f"{html.escape(loser)}")

    net.set_options("""
    {
      "physics": {"enabled": false},
      "interaction": {"hover": true, "navigationButtons": true,
                      "keyboard": true, "tooltipDelay": 80,
                      "hideEdgesOnDrag": true},
      "edges": {"smooth": false, "color": {"inherit": false},
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.45}}},
      "nodes": {"font": {"color": "#f0f0f0", "size": 13,
                         "strokeWidth": 3, "strokeColor": "#15181c"},
                "borderWidth": 1}
    }
    """)

    net.write_html(output_path, notebook=False, open_browser=False)
    # inject the legend overlay
    with open(output_path, "r", encoding="utf-8") as fh:
        page = fh.read()
    page = page.replace("<body>", "<body>\n" + _legend_html(), 1)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(page)

    abspath = os.path.abspath(output_path)
    if open_browser:
        webbrowser.open("file://" + abspath)
    return abspath


def _cli() -> None:
    import argparse

    from data import load_matches, team_confederations
    from elo_model import build_elo
    from graph_model import build_graph

    ap = argparse.ArgumentParser(
        description="Visualise the FIFA dominance graph.")
    ap.add_argument("--path", nargs=2, metavar=("TEAM_A", "TEAM_B"),
                    help="matplotlib view: highlight the reasoning path "
                         "between two teams")
    ap.add_argument("--save", metavar="FILE",
                    help="output path (PNG for --path, else HTML)")
    ap.add_argument("--no-open", action="store_true",
                    help="don't open the browser automatically")
    args = ap.parse_args()

    df = load_matches()
    gm = build_graph(df)

    if args.path:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        a, b = args.path
        pred = gm.predict(a, b)
        print(pred.get("explanation", ""))
        fig, ax = plt.subplots(figsize=(13, 8))
        draw_prediction(gm, pred, ax)
        fig.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=140, bbox_inches="tight")
            print(f"saved -> {args.save}")
        else:
            plt.show()
        return

    # default: the big interactive browser graph of every team
    elo = build_elo(df)
    confeds = team_confederations(df)
    out = args.save or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "fifa_graph.html")
    path = build_interactive(gm, elo, confeds, output_path=out,
                             open_browser=not args.no_open)
    print(f"interactive graph ({gm.graph.number_of_nodes()} teams, "
          f"{gm.graph.number_of_edges()} edges) -> {path}")


if __name__ == "__main__":
    _cli()
