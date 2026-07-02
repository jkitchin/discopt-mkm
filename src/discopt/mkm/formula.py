"""Chemical-formula parsing for species elemental composition.

Best-effort: standard formulas (digits, parentheses, the surface-site ``*``
stripped) parse correctly; non-formula labels (``A``, ``PO1``, ``OMP``) cannot be
disambiguated and should be given an explicit ``composition=`` instead.
"""

from __future__ import annotations

import re

# element symbols (1-118) for validating that a parse looks like a real formula
ELEMENTS = set(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
    Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg
    Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()
)

_TOKEN = re.compile(r"[A-Z][a-z]?|\(|\)|\d+")


def parse_formula(name: str) -> dict:
    """Parse a chemical formula into ``{element: count}``.

    Strips the surface-site marker ``*`` and supports parenthesized groups with a
    multiplier (e.g. ``Ca(OH)2``). Returns ``{}`` for an empty/site-only name.
    """
    s = name.replace("*", "").strip()
    if not s:
        return {}
    tokens = _TOKEN.findall(s)
    stack = [{}]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "(":
            stack.append({})
            i += 1
        elif t == ")":
            group = stack.pop()
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            for e, c in group.items():
                stack[-1][e] = stack[-1].get(e, 0) + c * mult
        elif t[0].isalpha():
            i += 1
            cnt = 1
            if i < len(tokens) and tokens[i].isdigit():
                cnt = int(tokens[i])
                i += 1
            stack[-1][t] = stack[-1].get(t, 0) + cnt
        else:  # a stray number (e.g. a config index after a stripped '-') -> ignore
            i += 1
    return {e: c for e, c in stack[0].items() if c}


def looks_like_formula(name: str) -> bool:
    """True if ``name`` parses into recognized element symbols (a real formula)."""
    parsed = parse_formula(name)
    return bool(parsed) and all(e in ELEMENTS for e in parsed)
