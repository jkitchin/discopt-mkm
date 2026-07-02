"""Figures: free-energy (reaction-coordinate) diagrams and reaction networks.

Needs matplotlib (and networkx for the network graph). Functions return a
matplotlib ``Axes`` so they compose with user figures, or accept ``ax=None`` to
make their own.
"""

from __future__ import annotations

import numpy as np

from discopt.mkm import numeric


def energy_diagram(mkm, T=None, ax=None, pressures=None):
    """Free-energy diagram along the mechanism (states and transition states).

    Follows the reactions in their listed order, accumulating the reaction free
    energy ``dG_j`` to place each state and the forward barrier ``Ea_j`` to place
    each transition state (Arrhenius steps; barrierless connector otherwise).
    Energies are standard free energies at ``T`` (model ``T`` by default); pass
    ``pressures={gas: P}`` to add the gas chemical potential ``R T ln P``.
    """
    import matplotlib.pyplot as plt

    T = mkm.T if T is None else float(T)
    theta0 = {a: 0.0 for a in mkm.adsorbates}  # clean-surface reference
    RT = mkm.R * T

    # cumulative state energies and transition-state heights
    states = [0.0]
    ts = []
    for rxn in mkm.reactions:
        dG = numeric.reaction_free_energy(mkm, rxn, T, theta0)
        if pressures:
            # gas chemical-potential correction on the reaction free energy
            for g, nu in rxn.net_stoich().items():
                if g in pressures and g in mkm.gas_species:
                    dG += nu * RT * float(np.log(max(pressures[g], 1e-300)))
        nxt = states[-1] + dG
        if not rxn.explicit_rate and not rxn.equilibrated:
            ts.append(states[-1] + rxn.Ea)  # forward barrier
        else:
            ts.append(max(states[-1], nxt))  # no explicit barrier
        states.append(nxt)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 + 1.1 * len(mkm.reactions), 4), layout="constrained")

    w = 0.3

    def _smoothstep(x0, y0, x1, y1, n=40):
        """Cubic Hermite with horizontal tangents at both ends (flat-to-flat)."""
        t = np.linspace(0.0, 1.0, n)
        return x0 + (x1 - x0) * t, y0 + (y1 - y0) * (3 * t**2 - 2 * t**3)

    for i, G in enumerate(states):
        ax.hlines(G, i - w, i + w, color="k", lw=2.5)
        ax.annotate(f"S{i}", (i, G), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    for j, (rxn, b) in enumerate(zip(mkm.reactions, ts)):
        x0, x1 = j + w, j + 1 - w
        if b > max(states[j], states[j + 1]) + 1e-9:
            # rise to the barrier peak then fall, smooth (flat) at plateaus and peak
            xl, yl = _smoothstep(x0, states[j], j + 0.5, b)
            xr, yr = _smoothstep(j + 0.5, b, x1, states[j + 1])
            ax.plot(np.concatenate([xl, xr]), np.concatenate([yl, yr]), color="0.5", lw=1.3)
        else:
            # barrierless connector: a single smooth step between the two states
            ax.plot(*_smoothstep(x0, states[j], x1, states[j + 1]), color="0.5", lw=1.3)
        if b > max(states[j], states[j + 1]) + 1e-9:
            ax.annotate(rxn.name.split(":")[0], (j + 0.5, b), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7, color="#a33")

    ax.set_xlabel("reaction coordinate")
    ax.set_ylabel(f"free energy (T = {T:g})")
    ax.set_title(f"{mkm.name}: free-energy diagram")
    ax.set_xticks([])
    return ax


def network_graph(mkm, solution=None, ax=None, seed=1):
    """Bipartite reaction network: species (circles) and reactions (squares).

    With a ``solution``, reaction nodes are sized/colored by the magnitude of the
    net rate of progress (the flux through each step).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for sp in mkm.species:
        G.add_node(("sp", sp), kind="species", label=sp.name)
    fluxes = {}
    for j, rxn in enumerate(mkm.reactions):
        rnode = ("rxn", j)
        G.add_node(rnode, kind="reaction", label=f"R{j + 1}")
        for s in rxn.reactants:
            G.add_edge(("sp", s), rnode)
        for s in rxn.products:
            G.add_edge(rnode, ("sp", s))
        if solution is not None:
            try:
                fluxes[rnode] = abs(solution.rate_of_progress(rxn))
            except Exception:
                fluxes[rnode] = 0.0

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    pos = nx.spring_layout(G, seed=seed, k=0.9)

    sp_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "species"]
    rx_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "reaction"]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.6", arrowsize=8, width=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=sp_nodes, node_color="#cfe3ff",
                           edgecolors="#345", node_shape="o", node_size=900, ax=ax)
    if solution is not None and fluxes:
        fmax = max(fluxes.values()) or 1.0
        sizes = [200 + 700 * fluxes[n] / fmax for n in rx_nodes]
        colors = [fluxes[n] / fmax for n in rx_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=rx_nodes, node_color=colors, cmap="OrRd",
                               node_shape="s", node_size=sizes, edgecolors="k", ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=rx_nodes, node_color="#ffd9b3",
                               node_shape="s", node_size=350, edgecolors="k", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: d["label"] for n, d in G.nodes(data=True)},
                            font_size=7, ax=ax)
    ax.set_title(f"{mkm.name}: reaction network" + (" (flux-weighted)" if solution is not None else ""))
    ax.axis("off")
    return ax


class _InlineHTML:
    """Tiny wrapper so an HTML string renders inline in Jupyter and can be saved."""

    def __init__(self, html: str):
        self.html = html

    def _repr_html_(self):
        return self.html

    def save(self, path: str):
        """Write a standalone ``.html`` file (open it in any browser)."""
        with open(path, "w") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
                    + self.html + "</body></html>")
        return path


def interactive_network(mkm, solution=None, height="540px", physics=True):
    """Interactive reaction-network graph (draggable nodes, hover tooltips).

    Renders a self-contained `vis-network <https://visjs.github.io/vis-network/>`_
    widget (loaded from a CDN, so viewing needs internet): species are circles,
    reactions are squares, and with a ``solution`` each **edge thickness scales
    with the net flux** (rate of progress) through that step. Hovering a node
    shows its details (coverage / partial pressure / rate constant and net rate).

    Returns an object that displays inline in Jupyter; call ``.save("net.html")``
    to write a standalone file. The static :func:`network_graph` (matplotlib) and
    :func:`to_dot` (Graphviz) remain available for non-interactive contexts.
    """
    import json

    gas = set(mkm.gas_species)
    ads = set(mkm.adsorbates)

    nodes, sp_id = [], {}
    for i, sp in enumerate(mkm.species):
        sp_id[sp] = f"s{i}"
        if sp in gas:
            color, group = "#cfe3ff", "gas"
            detail = "gas"
        elif sp in ads:
            color, group = "#cdeccd", "adsorbate"
            detail = "adsorbate"
        else:
            color, group = "#e0e0e0", "site"
            detail = "site"
        title = f"{sp.name}  ·  {detail}"
        if solution is not None:
            try:
                if sp in ads:
                    title += f"  ·  θ = {solution.coverage(sp):.3g}"
                elif sp not in gas:
                    title += f"  ·  free θ = {solution.free_coverage(sp):.3g}"
            except Exception:
                pass
        nodes.append({"id": sp_id[sp], "label": sp.name, "title": title,
                      "shape": "dot", "size": 16, "color": color, "group": group})

    edges, fluxes = [], []
    for j, rxn in enumerate(mkm.reactions):
        rid = f"r{j}"
        flux = None
        if solution is not None:
            try:
                flux = float(solution.rate_of_progress(rxn))
            except Exception:
                flux = None
        title = f"R{j + 1}: {rxn.name}  ·  {rxn.equation()}"
        if not rxn.equilibrated and not rxn.explicit_rate:
            title += f"  ·  A = {rxn.A:.3g}, Ea = {rxn.Ea:.3g}"
        if flux is not None:
            title += f"  ·  net rate = {flux:.3g}"
        nodes.append({"id": rid, "label": f"R{j + 1}", "title": title, "shape": "square",
                      "size": 14, "color": "#f4a259", "group": "reaction"})
        for s in rxn.reactants:
            edges.append({"from": sp_id[s], "to": rid, "flux": abs(flux) if flux is not None else None})
        for s in rxn.products:
            edges.append({"from": rid, "to": sp_id[s], "flux": abs(flux) if flux is not None else None})
        if flux is not None:
            fluxes.append(abs(flux))

    # map flux -> edge width (log-compressed: fluxes span orders of magnitude)
    fmax = max(fluxes) if fluxes else 0.0
    for e in edges:
        if e["flux"] and fmax > 0:
            e["width"] = 1.0 + 7.0 * (np.log1p(e["flux"]) / np.log1p(fmax))
            e["title"] = f"flux = {e['flux']:.3g}"
        else:
            e["width"] = 1.0
        e.pop("flux", None)
        e["arrows"] = "to"
        e["color"] = {"color": "#888", "opacity": 0.8}

    flux_note = "  —  edge thickness ∝ net flux" if fluxes else ""
    options = {
        "nodes": {"borderWidth": 1, "color": {"border": "#34506b"}, "font": {"size": 14}, "shadow": True},
        "edges": {"smooth": {"type": "dynamic"}},
        "interaction": {"hover": True, "dragNodes": True, "zoomView": True,
                        "dragView": True, "tooltipDelay": 80, "navigationButtons": True},
        "physics": {"enabled": bool(physics),
                    "barnesHut": {"springLength": 130, "avoidOverlap": 0.4},
                    "stabilization": {"iterations": 150}},
    }
    # A full standalone document with the vis-network script. It runs inside an
    # <iframe srcdoc>, because Jupyter / VS Code strip <script> tags from cell
    # HTML outputs for security -- an iframe has its own browsing context where
    # scripts do execute.
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<script src='https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js'></script>"
        "<style>html,body{margin:0;padding:0;height:100%}#net{width:100%;height:100%}</style>"
        "</head><body><div id='net'></div><script>"
        f"var nodes=new vis.DataSet({json.dumps(nodes)});"
        f"var edges=new vis.DataSet({json.dumps(edges)});"
        f"new vis.Network(document.getElementById('net'),{{nodes:nodes,edges:edges}},{json.dumps(options)});"
        "</script></body></html>"
    )
    srcdoc = doc.replace("&", "&amp;").replace('"', "&quot;")
    html = (
        "<div style='font-family:sans-serif'>"
        f"<div style='font-weight:600;margin:4px 2px'>{mkm.name}: reaction network{flux_note}</div>"
        f"<iframe srcdoc=\"{srcdoc}\" style='width:100%;height:{height};"
        "border:1px solid #ddd;border-radius:6px'></iframe></div>"
    )
    return _InlineHTML(html)


def to_dot(mkm, solution=None) -> str:
    """Graphviz DOT source for the bipartite reaction network (no dependencies)."""
    lines = [f'digraph "{mkm.name}" {{', "  rankdir=LR;", "  node [fontsize=10];"]
    for sp in mkm.species:
        lines.append(f'  "{sp.name}" [shape=ellipse, style=filled, fillcolor="#cfe3ff"];')
    for j, rxn in enumerate(mkm.reactions):
        label = f"R{j + 1}"
        if solution is not None:
            try:
                label += f"\\n{solution.rate_of_progress(rxn):.2g}"
            except Exception:
                pass
        lines.append(f'  "R{j + 1}" [shape=box, style=filled, fillcolor="#ffd9b3", label="{label}"];')
        for s in rxn.reactants:
            lines.append(f'  "{s.name}" -> "R{j + 1}";')
        for s in rxn.products:
            lines.append(f'  "R{j + 1}" -> "{s.name}";')
    lines.append("}")
    return "\n".join(lines)
