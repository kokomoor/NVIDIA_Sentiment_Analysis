"""§9.1 — InvestorBranch aggregator tests."""

from __future__ import annotations

import math

from nvda_sentiment.config import INVESTOR_COMPONENT_WEIGHTS, Settings
from nvda_sentiment.features.investor_branch import InvestorBranch
from nvda_sentiment.schemas import InvestorSubScore


class _FakeAdapter:
    def __init__(self, name: str, score: float, ok: bool):
        self._sub = InvestorSubScore(name=name, score=score, ok=ok, detail={})
        self.use_cache = True

    def get_signal(self) -> InvestorSubScore:
        return self._sub


class _RaisingAdapter:
    def __init__(self, name: str):
        self._name = name
        self.use_cache = True

    def get_signal(self):
        raise RuntimeError(f"{self._name} blew up")


def _make_branch(
    *,
    options_score=0.0, options_ok=True,
    shorts_score=0.0, shorts_ok=True,
    analyst_score=0.0, analyst_ok=True,
    social_score=0.0, social_ok=True,
    broad_score=0.0, broad_ok=True,
    tmp_path=None,
) -> InvestorBranch:
    settings = Settings(cache_dir=tmp_path) if tmp_path is not None else Settings()
    return InvestorBranch(
        settings,
        options=_FakeAdapter("options_flow", options_score, options_ok),
        shorts=_FakeAdapter("short_interest", shorts_score, shorts_ok),
        analyst=_FakeAdapter("analyst_signal", analyst_score, analyst_ok),
        social=_FakeAdapter("social", social_score, social_ok),
        broad_market=_FakeAdapter("broad_market", broad_score, broad_ok),
    )


def test_all_five_ok_uses_full_weights(tmp_path) -> None:
    branch = _make_branch(
        options_score=0.5,
        shorts_score=-0.2,
        analyst_score=0.3,
        social_score=0.1,
        broad_score=-0.4,
        tmp_path=tmp_path,
    )
    out = branch.run(include_broad_market=True)
    w = INVESTOR_COMPONENT_WEIGHTS
    expected = (
        w["options_flow"] * 0.5
        + w["short_interest"] * -0.2
        + w["analyst_signal"] * 0.3
        + w["social"] * 0.1
        + w["broad_market"] * -0.4
    )
    assert math.isclose(out.investor_component, expected, abs_tol=1e-9)
    assert out.ok is True
    assert len(out.sub_scores) == 5
    assert math.isclose(out.investor_score, 50.0 + 50.0 * expected, abs_tol=1e-9)


def test_two_fail_three_ok_redistributes_weights(tmp_path) -> None:
    # options=0.8 ok, shorts fail, analyst=0.2 ok, social fail, broad=-0.6 ok
    branch = _make_branch(
        options_score=0.8, options_ok=True,
        shorts_score=0.9, shorts_ok=False,
        analyst_score=0.2, analyst_ok=True,
        social_score=-0.5, social_ok=False,
        broad_score=-0.6, broad_ok=True,
        tmp_path=tmp_path,
    )
    out = branch.run(include_broad_market=True)
    w = INVESTOR_COMPONENT_WEIGHTS
    usable_total = w["options_flow"] + w["analyst_signal"] + w["broad_market"]
    expected = (
        w["options_flow"] * 0.8
        + w["analyst_signal"] * 0.2
        + w["broad_market"] * -0.6
    ) / usable_total
    assert math.isclose(out.investor_component, expected, abs_tol=1e-9)
    assert out.ok is True


def test_all_fail_returns_component_zero_and_not_ok(tmp_path) -> None:
    branch = _make_branch(
        options_ok=False, shorts_ok=False, analyst_ok=False,
        social_ok=False, broad_ok=False,
        tmp_path=tmp_path,
    )
    out = branch.run(include_broad_market=True)
    assert out.investor_component == 0.0
    assert out.ok is False
    assert math.isclose(out.investor_score, 50.0, abs_tol=1e-9)


def test_include_broad_market_false_renormalizes_over_four(tmp_path) -> None:
    branch = _make_branch(
        options_score=0.4,
        shorts_score=0.0,
        analyst_score=0.2,
        social_score=-0.1,
        broad_score=0.9,     # should be EXCLUDED
        broad_ok=True,
        tmp_path=tmp_path,
    )
    out = branch.run(include_broad_market=False)
    assert len(out.sub_scores) == 4
    assert all(s.name != "broad_market" for s in out.sub_scores)
    w = INVESTOR_COMPONENT_WEIGHTS
    usable_total = (
        w["options_flow"] + w["short_interest"] + w["analyst_signal"] + w["social"]
    )
    expected = (
        w["options_flow"] * 0.4
        + w["short_interest"] * 0.0
        + w["analyst_signal"] * 0.2
        + w["social"] * -0.1
    ) / usable_total
    assert math.isclose(out.investor_component, expected, abs_tol=1e-9)


def test_raising_adapter_is_degraded_to_ok_false(tmp_path) -> None:
    settings = Settings(cache_dir=tmp_path)
    branch = InvestorBranch(
        settings,
        options=_RaisingAdapter("options_flow"),
        shorts=_FakeAdapter("short_interest", -0.5, True),
        analyst=_FakeAdapter("analyst_signal", 0.2, True),
        social=_FakeAdapter("social", 0.0, True),
        broad_market=_FakeAdapter("broad_market", 0.1, True),
    )
    out = branch.run(include_broad_market=True)
    options_sub = [s for s in out.sub_scores if s.name == "options_flow"][0]
    assert options_sub.ok is False
    assert options_sub.score == 0.0
    # Branch still reports ok because other subs are ok.
    assert out.ok is True


def test_set_use_cache_propagates(tmp_path) -> None:
    branch = _make_branch(tmp_path=tmp_path)
    branch.set_use_cache(False)
    for adapter in (branch.options, branch.shorts, branch.analyst, branch.social, branch.broad_market):
        assert adapter.use_cache is False
    branch.set_use_cache(True)
    for adapter in (branch.options, branch.shorts, branch.analyst, branch.social, branch.broad_market):
        assert adapter.use_cache is True
