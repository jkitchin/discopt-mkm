"""LaTeX / HTML rendering of species, reactions, mechanisms, and solutions.

Pure formatting: takes domain objects and returns strings. The classes attach
thin ``to_latex`` / ``to_html`` / ``_repr_html_`` / ``_repr_latex_`` methods that
delegate here, so mechanisms render automatically in Jupyter and export cleanly
to documents.
"""

from __future__ import annotations

import re

# arrows per reaction type
_ARROW_LATEX = {"rev": r"\rightleftharpoons", "irr": r"\rightarrow", "eq": r"\mathrel{\underset{\mathrm{eq}}{\rightleftharpoons}}"}
_ARROW_HTML = {"rev": "&#8652;", "irr": "&rarr;", "eq": "&#8652;<sub>eq</sub>"}


def _kind(rxn) -> str:
    if rxn.equilibrated:
        return "eq"
    if rxn.irreversible:
        return "irr"
    return "rev"


def _coeff(c: float) -> str:
    ci = int(round(c))
    return "" if abs(c - 1.0) < 1e-9 else (str(ci) if abs(c - ci) < 1e-9 else f"{c:g}")


# -- LaTeX -------------------------------------------------------------------
def species_latex(name: str) -> str:
    """Format a species formula as math: digits subscripted, ``*`` as a superscript star."""
    s = re.sub(r"([A-Za-z\)\]])(\d+)", r"\1_{\2}", name)
    s = s.replace("*", r"{}^{\ast}")
    return s


def _side_latex(stoich: dict) -> str:
    parts = []
    for sp, c in stoich.items():
        co = _coeff(c)
        parts.append((co + r"\," if co else "") + species_latex(sp.name))
    return " + ".join(parts) if parts else r"\varnothing"


def reaction_latex(rxn, inline: bool = True) -> str:
    body = f"{_side_latex(rxn.reactants)} {_ARROW_LATEX[_kind(rxn)]} {_side_latex(rxn.products)}"
    return f"${body}$" if inline else body


def mechanism_latex(mkm) -> str:
    """An ``align`` block of all steps (one per line, arrows aligned)."""
    rows = []
    for rxn in mkm.reactions:
        lhs = _side_latex(rxn.reactants)
        rhs = f"{_ARROW_LATEX[_kind(rxn)]} {_side_latex(rxn.products)}"
        rows.append(f"  {lhs} &{rhs}")
    return "\\begin{align}\n" + " \\\\\n".join(rows) + "\n\\end{align}"


# -- HTML --------------------------------------------------------------------
def species_html(name: str) -> str:
    s = re.sub(r"([A-Za-z\)\]])(\d+)", r"\1<sub>\2</sub>", name)
    s = s.replace("*", "<sup>&lowast;</sup>")
    return s


def _side_html(stoich: dict) -> str:
    parts = []
    for sp, c in stoich.items():
        co = _coeff(c)
        parts.append((co + "&#8201;" if co else "") + species_html(sp.name))
    return " + ".join(parts) if parts else "&empty;"


def reaction_html(rxn) -> str:
    a = _ARROW_HTML[_kind(rxn)]
    return f"{_side_html(rxn.reactants)} {a} {_side_html(rxn.products)}"


def _kinetics_html(rxn) -> str:
    if rxn.equilibrated:
        return f"equilibrated, K<sub>eq</sub>={rxn.Keq:g}" if rxn.Keq is not None else "equilibrated (thermo)"
    if rxn.explicit_rate:
        keq = "&mdash;" if rxn.irreversible else f"{rxn.Keq:g}"
        return f"k<sub>f</sub>={rxn.kf:g}, K<sub>eq</sub>={keq}"
    return f"A={rxn.A:g}, E<sub>a</sub>={rxn.Ea:g}"


def mechanism_html(mkm) -> str:
    """A mechanism table: step, reaction, kinetics, type."""
    rows = [
        "<table style='border-collapse:collapse'>",
        "<tr><th style='text-align:right'>#</th><th style='text-align:left'>reaction</th>"
        "<th style='text-align:left'>kinetics</th><th>type</th></tr>",
    ]
    type_label = {"rev": "reversible", "irr": "irreversible", "eq": "quasi-equilibrium"}
    for i, rxn in enumerate(mkm.reactions, 1):
        rows.append(
            f"<tr><td style='text-align:right'>{i}</td>"
            f"<td style='text-align:left'>{reaction_html(rxn)}</td>"
            f"<td style='text-align:left'>{_kinetics_html(rxn)}</td>"
            f"<td style='text-align:center'>{type_label[_kind(rxn)]}</td></tr>"
        )
    rows.append("</table>")
    title = f"<b>{mkm.name}</b> &mdash; T={mkm.T:g}, {len(mkm.species)} species, {len(mkm.reactions)} steps<br>"
    return title + "\n".join(rows)


def solution_html(sol) -> str:
    """Coverages and net rates of progress at the solved steady state."""
    mkm = sol.mkm
    cov = ["<b>coverages</b><table style='border-collapse:collapse'>"]
    for a in mkm.adsorbates:
        cov.append(f"<tr><td>&theta;[{species_html(a.name)}]</td><td style='text-align:right'>{sol.coverage(a):.4g}</td></tr>")
    for s in mkm.sites:
        cov.append(f"<tr><td>&theta;[{species_html(s.name)}<sub>free</sub>]</td><td style='text-align:right'>{sol.free_coverage(s):.4g}</td></tr>")
    cov.append("</table>")

    rates = ["<b>rates of progress</b><table style='border-collapse:collapse'>"]
    for rxn in mkm.reactions:
        rates.append(f"<tr><td style='text-align:left'>{reaction_html(rxn)}</td><td style='text-align:right'>{sol.rate_of_progress(rxn):.4g}</td></tr>")
    rates.append("</table>")
    return (
        f"<b>{mkm.name}</b> steady state (status: {sol.status})<br>"
        "<div style='display:flex;gap:2em'>"
        f"<div>{''.join(cov)}</div><div>{''.join(rates)}</div></div>"
    )
