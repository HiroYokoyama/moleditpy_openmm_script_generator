"""Graph automorphisms of a molecule type (elements + bonds).

OpenMM matches a residue to its template by elements and bonds only, so for
chemically equivalent atoms (the three hydrogens of a methyl group, say) it may
pick any of the equivalent mappings.  Per-atom parameters are reproduced
exactly only when they are invariant under those automorphisms; this module
enumerates them so the converter can detect when they are not.
"""

from __future__ import annotations

from collections.abc import Sequence


def _refine(elements: Sequence[str], neighbours: list[list[int]]) -> list[int]:
    colors = [hash((e, len(nb))) for e, nb in zip(elements, neighbours)]
    for _ in range(len(elements)):
        signatures = [
            (colors[i], tuple(sorted(colors[j] for j in neighbours[i]))) for i in range(len(elements))
        ]
        palette: dict[tuple, int] = {}
        new = [palette.setdefault(sig, len(palette)) for sig in signatures]
        if len(set(new)) == len(set(colors)):
            return new
        colors = new
    return colors


def automorphisms(
    elements: Sequence[str], bonds: Sequence[tuple[int, int]], limit: int = 5000
) -> tuple[list[tuple[int, ...]], bool]:
    """All automorphisms (as permutations), and whether the list is complete.

    Enumeration stops after ``limit`` permutations; the result is then marked
    incomplete and callers should assume any same-colour atoms may be swapped.
    """
    n = len(elements)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        if i != j:
            neighbours[i].append(j)
            neighbours[j].append(i)
    adjacency = [set(nb) for nb in neighbours]
    colors = _refine(elements, neighbours)
    by_color: dict[int, list[int]] = {}
    for atom, color in enumerate(colors):
        by_color.setdefault(color, []).append(atom)

    # Visit atoms breadth-first so each new atom is usually adjacent to a mapped one.
    order: list[int] = []
    seen = set()
    for start in sorted(range(n), key=lambda a: (len(by_color[colors[a]]), a)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            atom = queue.pop(0)
            order.append(atom)
            for nb in neighbours[atom]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)

    results: list[tuple[int, ...]] = []
    mapping = [-1] * n
    used = [False] * n
    complete = True

    def extend(depth: int) -> bool:
        nonlocal complete
        if depth == n:
            results.append(tuple(mapping))
            if len(results) >= limit:
                complete = False
                return False
            return True
        atom = order[depth]
        for candidate in by_color[colors[atom]]:
            if used[candidate]:
                continue
            ok = True
            for nb in neighbours[atom]:
                image = mapping[nb]
                if image >= 0 and image not in adjacency[candidate]:
                    ok = False
                    break
            if not ok:
                continue
            mapping[atom] = candidate
            used[candidate] = True
            if not extend(depth + 1):
                return False
            mapping[atom] = -1
            used[candidate] = False
        return True

    extend(0)
    return results, complete


def equivalence_classes(n: int, perms: Sequence[tuple[int, ...]]) -> list[int]:
    """Orbit id per atom under the given permutations."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for perm in perms:
        for a, b in enumerate(perm):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return [find(a) for a in range(n)]
