"""Self-contained HTML mechanism report (table + structure + figures + solution)."""

from __future__ import annotations

import base64
import io


def _fig_img(ax) -> str:
    """Embed a matplotlib Axes' figure as an inline base64 SVG ``<img>``."""
    import matplotlib.pyplot as plt

    fig = ax.get_figure()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue().encode()).decode()
    return f'<img style="max-width:100%" src="data:image/svg+xml;base64,{b64}"/>'


def report_html(mkm, solution=None, target=None, figures=True) -> str:
    """A self-contained HTML report for a mechanism (and optionally a solution).

    Bundles the mechanism table, stoichiometric structure (routes, independence,
    conservation laws), the free-energy diagram and reaction-network figures, and
    — if ``solution`` is given — the steady state, plus degree of rate control and
    apparent orders / activation energy for ``target`` (a species).
    """
    from discopt.mkm import render, viz
    from discopt.mkm.analysis import stoichiometry as st

    p = [f"<h2>{mkm.name}</h2>", "<h3>Mechanism</h3>", render.mechanism_html(mkm)]

    p.append("<h3>Stoichiometric structure</h3><ul>")
    p.append(f"<li>independent reactions: {st.n_independent_reactions(mkm)} of {len(mkm.reactions)}</li>")
    p.append(f"<li>conserved quantities: {st.n_conservation_laws(mkm)}</li>")
    for _, overall in st.reaction_routes(mkm):
        terms = " ".join(f"{v:+g}&#8201;{render.species_html(s.name)}" for s, v in overall.items())
        p.append(f"<li>overall reaction: {terms or 'closed cycle'}</li>")
    for law in st.site_conservation_laws(mkm):
        p.append("<li>site balance: " + " + ".join(render.species_html(s.name) for s in law) + " = const</li>")
    p.append("</ul>")

    if figures:
        p.append("<h3>Free-energy diagram</h3>")
        p.append(_fig_img(viz.energy_diagram(mkm)))
        p.append("<h3>Reaction network</h3>")
        p.append(_fig_img(viz.network_graph(mkm, solution=solution)))

    if solution is not None:
        p.append("<h3>Steady state</h3>")
        p.append(render.solution_html(solution))
        if target is not None:
            from discopt.mkm.analysis import (
                apparent_activation_energy,
                apparent_orders,
                degree_of_rate_control,
            )

            try:
                X = degree_of_rate_control(solution, species=target)
                rows = "".join(
                    f"<tr><td style='text-align:left'>{render.reaction_html(r)}</td>"
                    f"<td style='text-align:right'>{x:+.3f}</td></tr>"
                    for r, x in X.items()
                )
                p.append(f"<h3>Degree of rate control ({render.species_html(target.name)})</h3>"
                         f"<table style='border-collapse:collapse'>{rows}</table>")
            except Exception as e:
                p.append(f"<p><i>DRC unavailable: {e}</i></p>")
            try:
                orders = apparent_orders(solution, target)
                ea = apparent_activation_energy(solution, target)
                ords = ", ".join(f"{render.species_html(g.name)}: {n:+.2f}" for g, n in orders.items())
                p.append(f"<p><b>apparent orders</b>: {ords}<br><b>apparent E<sub>a</sub></b>: {ea:.3g}</p>")
            except Exception:
                pass

    return "<div style='font-family:sans-serif;max-width:46em'>" + "\n".join(p) + "</div>"


def write_report(mkm, path, solution=None, target=None, figures=True) -> str:
    """Write :func:`report_html` to a standalone HTML file; returns the path."""
    html = report_html(mkm, solution=solution, target=target, figures=figures)
    with open(path, "w") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'></head><body>{html}</body></html>")
    return path
