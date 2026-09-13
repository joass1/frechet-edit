"""Hand-computed tests for the retrieval metrics.

None of these involve FED. Every expected value is worked out by hand in a
comment, so the metric code is checked against arithmetic rather than against
itself.
"""

from __future__ import annotations

import math

import pytest

from experiments.metrics import (
    Scored,
    aggregate,
    evaluate_query,
    group_scores,
    hit_at_k,
    reciprocal_rank,
)


def S(pairs):
    return [Scored(t, s) for t, s in pairs]


class TestGrouping:
    def test_finite_scores_ascend_and_ties_group(self):
        groups = group_scores(S([("a", 2.0), ("b", 1.0), ("c", 2.0), ("d", 0.5)]))
        assert groups == [(0.5, ["d"]), (1.0, ["b"]), (2.0, ["a", "c"])]

    def test_infeasible_after_finite_and_ambiguous_last(self):
        groups = group_scores(
            S([("a", math.inf), ("b", 1.0), ("c", None), ("d", math.inf)])
        )
        assert groups == [(1.0, ["b"]), (math.inf, ["a", "d"]), (None, ["c"])]


class TestHitAtK:
    def test_clean_first_place(self):
        groups = group_scores(S([("t1", 0.0), ("t2", 1.0), ("t3", 2.0)]))
        assert hit_at_k(groups, {"t1"}, 1) == (1.0, 1.0, 1.0)

    def test_relevant_at_rank_three_misses_at_k1(self):
        groups = group_scores(S([("t1", 0.0), ("t2", 1.0), ("t3", 2.0)]))
        assert hit_at_k(groups, {"t3"}, 1) == (0.0, 0.0, 0.0)
        assert hit_at_k(groups, {"t3"}, 5) == (1.0, 1.0, 1.0)

    def test_full_tie_does_not_score_one_strictly(self):
        """Four templates all tied at 3.0, one relevant.

        Strict: the relevant one is placed last, rank 4, so Hit@1 = 0.
        Optimistic: rank 1, Hit@1 = 1.
        Expected: P(relevant first) = 1/4 = 0.25.
        """
        groups = group_scores(S([("a", 3.0), ("b", 3.0), ("c", 3.0), ("d", 3.0)]))
        strict, expected, optimistic = hit_at_k(groups, {"c"}, 1)
        assert strict == 0.0
        assert optimistic == 1.0
        assert expected == pytest.approx(0.25)

    def test_tie_group_with_two_relevant(self):
        """Four tied, two relevant, k = 1.

        Expected = 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5.
        Strict: first relevant at rank 4-2+1 = 3 > 1, so 0.
        """
        groups = group_scores(S([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]))
        strict, expected, optimistic = hit_at_k(groups, {"a", "b"}, 1)
        assert (strict, optimistic) == (0.0, 1.0)
        assert expected == pytest.approx(0.5)

    def test_tie_group_partially_inside_k(self):
        """One template better, then three tied with one relevant, k = 2.

        before = 1, size = 3, rel = 1, t = k - before = 1.
        Expected = 1 - C(2,1)/C(3,1) = 1 - 2/3 = 1/3.
        """
        groups = group_scores(S([("x", 0.0), ("a", 1.0), ("b", 1.0), ("c", 1.0)]))
        strict, expected, optimistic = hit_at_k(groups, {"b"}, 2)
        assert strict == 0.0 and optimistic == 1.0
        assert expected == pytest.approx(1.0 / 3.0)

    def test_whole_tie_group_inside_k_is_certain(self):
        groups = group_scores(S([("a", 1.0), ("b", 1.0), ("c", 5.0)]))
        assert hit_at_k(groups, {"b"}, 2) == (1.0, 1.0, 1.0)

    def test_relevant_only_in_the_ambiguous_group_is_never_retrieved(self):
        groups = group_scores(S([("a", 1.0), ("rel", None)]))
        assert hit_at_k(groups, {"rel"}, 5) == (0.0, 0.0, 0.0)

    def test_relevant_absent_from_gallery(self):
        groups = group_scores(S([("a", 1.0), ("b", 2.0)]))
        assert hit_at_k(groups, {"missing"}, 5) == (0.0, 0.0, 0.0)


class TestReciprocalRank:
    def test_rank_one(self):
        groups = group_scores(S([("t1", 0.0), ("t2", 1.0)]))
        assert reciprocal_rank(groups, {"t1"}) == (1.0, 1.0, 1.0)

    def test_rank_three(self):
        groups = group_scores(S([("a", 0.0), ("b", 1.0), ("c", 2.0)]))
        strict, expected, optimistic = reciprocal_rank(groups, {"c"})
        assert strict == pytest.approx(1 / 3)
        assert expected == pytest.approx(1 / 3)
        assert optimistic == pytest.approx(1 / 3)

    def test_three_way_tie_single_relevant(self):
        """Three tied, one relevant.

        P(first relevant at p) = C(3-p, 0)/C(3,1) = 1/3 for p = 1, 2, 3.
        E[1/rank] = (1/3)(1/1 + 1/2 + 1/3) = (1/3)(11/6) = 11/18.
        """
        groups = group_scores(S([("a", 2.0), ("b", 2.0), ("c", 2.0)]))
        strict, expected, optimistic = reciprocal_rank(groups, {"a"})
        assert strict == pytest.approx(1 / 3)
        assert optimistic == pytest.approx(1.0)
        assert expected == pytest.approx(11 / 18)

    def test_tie_with_two_relevant_of_three(self):
        """Three tied, two relevant.

        P(first relevant at p) = C(3-p, 1)/C(3,2): p=1 -> 2/3, p=2 -> 1/3.
        E[1/rank] = (2/3)(1) + (1/3)(1/2) = 5/6.
        """
        groups = group_scores(S([("a", 1.0), ("b", 1.0), ("c", 1.0)]))
        _, expected, _ = reciprocal_rank(groups, {"a", "b"})
        assert expected == pytest.approx(5 / 6)

    def test_offset_tie_group(self):
        """Two better templates, then two tied with one relevant.

        before = 2, size = 2, rel = 1: p = 1 or 2 with probability 1/2 each.
        E[1/rank] = (1/2)(1/3) + (1/2)(1/4) = 7/24.
        """
        groups = group_scores(S([("x", 0.0), ("y", 1.0), ("a", 2.0), ("b", 2.0)]))
        strict, expected, optimistic = reciprocal_rank(groups, {"a"})
        assert optimistic == pytest.approx(1 / 3)
        assert strict == pytest.approx(1 / 4)
        assert expected == pytest.approx(7 / 24)


class TestAbstention:
    def test_all_infeasible_gallery_is_an_abstention_not_a_hit(self):
        outcome = evaluate_query("q", S([("a", math.inf), ("b", math.inf)]), {"a"})
        assert outcome.abstained is True
        assert outcome.hit_at[1] == (0.0, 0.0, 0.0)
        assert outcome.rr == (0.0, 0.0, 0.0)
        assert outcome.top1_id is None

    def test_all_ambiguous_gallery_is_an_abstention(self):
        outcome = evaluate_query("q", S([("a", None), ("b", None)]), {"a"})
        assert outcome.abstained is True
        assert outcome.n_ambiguous == 2

    def test_abstentions_stay_in_the_denominator(self):
        """Two queries, one perfect and one abstained.

        Failure-aware Recall@1 = 1/2 = 0.5. Successful-only = 1/1 = 1.0.
        """
        good = evaluate_query("q1", S([("a", 0.0), ("b", 1.0)]), {"a"})
        bad = evaluate_query("q2", S([("a", math.inf), ("b", math.inf)]), {"a"})
        agg = aggregate("m", [good, bad], ks=(1,))
        assert agg.n_queries == 2 and agg.n_abstentions == 1
        assert agg.recall_at[1][1] == pytest.approx(0.5)
        assert agg.recall_at_successful[1][1] == pytest.approx(1.0)
        assert agg.coverage == pytest.approx(0.5)

    def test_infeasible_templates_rank_last_but_do_not_abstain(self):
        outcome = evaluate_query("q", S([("a", math.inf), ("b", 2.0)]), {"b"})
        assert outcome.abstained is False
        assert outcome.hit_at[1] == (1.0, 1.0, 1.0)


class TestOpenSet:
    def test_no_match_query_counts_as_a_false_match_when_accepted(self):
        openq = evaluate_query("neg", S([("a", 0.5), ("b", 2.0)]), relevant=set())
        agg = aggregate("m", [openq], ks=(1,), acceptance_threshold=1.0)
        assert agg.n_open_set == 1
        assert agg.n_false_matches == 1
        assert agg.false_match_rate == pytest.approx(1.0)

    def test_no_match_query_rejected_above_threshold(self):
        openq = evaluate_query("neg", S([("a", 5.0)]), relevant=set())
        agg = aggregate("m", [openq], ks=(1,), acceptance_threshold=1.0)
        assert agg.n_false_matches == 0
        assert agg.false_match_rate == pytest.approx(0.0)

    def test_open_set_queries_do_not_pollute_recall(self):
        closed = evaluate_query("q1", S([("a", 0.0), ("b", 1.0)]), {"a"})
        openq = evaluate_query("neg", S([("a", 0.5)]), relevant=set())
        agg = aggregate("m", [closed, openq], ks=(1,), acceptance_threshold=1.0)
        assert agg.n_queries == 1  # only the closed-set query
        assert agg.recall_at[1][1] == pytest.approx(1.0)

    def test_false_match_rate_is_none_when_no_open_set_queries_posed(self):
        closed = evaluate_query("q1", S([("a", 0.0)]), {"a"})
        agg = aggregate("m", [closed], ks=(1,))
        assert agg.n_open_set == 0
        assert agg.false_match_rate is None


class TestMultiPositive:
    def test_multiple_relevant_templates_use_the_first(self):
        groups = group_scores(S([("a", 0.0), ("b", 1.0), ("c", 2.0)]))
        assert reciprocal_rank(groups, {"b", "c"})[1] == pytest.approx(0.5)

    def test_hit_at_k_is_at_least_one_relevant_not_all(self):
        groups = group_scores(S([("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 3.0)]))
        # Two relevant, only one inside the top 2: Hit@2 is still 1.
        assert hit_at_k(groups, {"b", "d"}, 2) == (1.0, 1.0, 1.0)


class TestOrderingCannotUseTheTruth:
    def test_tied_group_result_is_independent_of_relevant_label(self):
        """Swapping which tied template is relevant must not change the numbers."""
        scored = S([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        first = hit_at_k(group_scores(scored), {"a"}, 1)
        second = hit_at_k(group_scores(scored), {"c"}, 1)
        assert first == second

    def test_strict_bound_never_exceeds_expected_never_exceeds_optimistic(self):
        scored = S([("a", 1.0), ("b", 1.0), ("c", 2.0), ("d", math.inf)])
        for rel in ({"a"}, {"b"}, {"c"}, {"a", "c"}):
            for k in (1, 2, 3):
                s, e, o = hit_at_k(group_scores(scored), rel, k)
                assert s <= e <= o, (rel, k, s, e, o)
            s, e, o = reciprocal_rank(group_scores(scored), rel)
            assert s <= e <= o, (rel, s, e, o)
