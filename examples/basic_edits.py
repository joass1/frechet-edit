"""Worked examples of the three edit modes, with independently verified witnesses.

Runs offline in a second, uses only the installed public API, and prints the
actual numbers rather than asserting anything it does not show.

    python examples/basic_edits.py
"""

from __future__ import annotations

import numpy as np

from frechet_edit import (
    discrete_edit_distance,
    ordinary_discrete_frechet,
    verify_witness,
)


def show(title: str, reference, observation, delta: float, operations: str) -> None:
    reference = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    observation = np.asarray(observation, dtype=np.float64).reshape(len(observation), -1)

    result = discrete_edit_distance(
        reference, observation, delta, operations=operations, return_witness=True
    )

    print(f"\n{title}")
    print("-" * len(title))
    print(f"  reference   : {reference.ravel().tolist()}")
    print(f"  observation : {observation.ravel().tolist()}")
    print(f"  delta       : {delta}   mode: {operations}")
    print(f"  ordinary discrete Frechet: {ordinary_discrete_frechet(reference, observation):.4g}")
    print(f"  status      : {result.status}   cost: {result.cost}")

    if result.witness_status != "certified":
        print(f"  witness     : {result.witness_status} ({result.detail})")
        return

    for edit in result.edits:
        payload = edit.as_dict()
        if payload["op"] == "delete":
            print(f"  edit        : delete original index {payload['index']}")
        else:
            point = [round(c, 6) for c in payload["point"]]
            print(
                f"  edit        : insert {point} in gap {payload['gap']} "
                f"(order {payload['order']})"
            )
    print(f"  edited curve: {result.edited_curve.ravel().tolist()}")
    report = verify_witness(reference, observation, result)
    print(f"  verified    : {report.ok}  (independent Frechet on the edited curve: "
          f"{report.ordinary_frechet:.4g})")


def main() -> None:
    print("frechet-edit: worked examples")
    print("=" * 40)
    print("The cost is a COUNT OF EDITS, never a distance in coordinate units.")

    show(
        "1. One isolated spike is one deletion",
        [0.0, 1.0, 2.0], [0.0, 1.0, 100.0, 2.0], 0.1, "delete",
    )

    show(
        "2. One inserted point covers TWO reference vertices",
        [0.0, 2.0, 4.0], [0.0], 1.0, "insert",
    )
    print("     the inserted point is 3.0 - a vertex of neither curve.")
    print("     Restricting insertion to reference vertices would cost 2.")

    show(
        "3. Insertion alone cannot remove a spike: infeasible, not an error",
        [0.0, 1.0, 2.0], [0.0, 1.0, 100.0, 2.0], 0.1, "insert",
    )

    show(
        "4. Mixed mode can delete everything and rebuild",
        [0.0, 10.0], [100.0, 200.0], 1.0, "both",
    )

    show(
        "5. Repeated vertices cost nothing: this is not string edit distance",
        [0.0, 0.0, 0.0], [0.0, 0.0], 0.5, "both",
    )

    print("\n6. Why the edit objective can rank differently from ordinary Frechet")
    print("-" * 68)
    reference = np.array([[0.0], [1.0], [2.0]])
    noisy = np.array([[0.0], [1.0], [100.0], [2.0]])  # right route, one bad sample
    wrong = np.array([[0.0], [1.0], [3.0]])           # different route, clean
    for name, candidate in (("noisy-but-correct", noisy), ("clean-but-wrong", wrong)):
        ordinary = ordinary_discrete_frechet(reference, candidate)
        cost = discrete_edit_distance(reference, candidate, 0.1, operations="both").cost
        print(f"  {name:20} ordinary Frechet = {ordinary:7.4g}   FED = {cost}")
    print("  Ordinary Frechet prefers the wrong candidate; FED prefers the noisy one.")
    print("  This demonstrates the MECHANISM. It is not evidence about real data;")
    print("  see docs/limitations.md.")


if __name__ == "__main__":
    main()
