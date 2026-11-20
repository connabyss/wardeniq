"""The business fallback tops the suite up to its floor instead of rewriting every rule (#48).

The fallback used to ask for one test per requirement rule, whatever the combined E2E
call had already written. It re-tested covered rules and wrote past the floor, and
_budget_suites() then trimmed the extra. These tests pin the top-up behaviour through
the real pipeline, with no external services.
"""
from testgen.prompt_builder import build_business_test_prompt
from testgen.service import generate_fresh_testcases_pipeline
from tests.test_generation_pipeline import FakeEmbedder, FakeStore, SpyLLM

FALLBACK_MARKER = "Generate one business rule test per requirement rule"
RULES = [
    "Only authenticated customers may create orders",
    "An order must contain at least one item",
    "Only one promo code may be applied per order",
    "Orders above 10 items need approval",
    "A cancelled order cannot be edited",
]


class FiveRuleStore(FakeStore):
    def build_unified_context(self, feature_id, version_number):
        context = super().build_unified_context(feature_id, version_number)
        context["businessContext"] = {"prd": {"requirements": list(RULES)}}
        return context


class OneBusinessTestLLM(SpyLLM):
    """The combined E2E call returns one business test, so the fallback has to top up."""

    def chat_json(self, system, user, **kwargs):
        if "Generate E2E flows, edge cases, and business rule tests" in user:
            return {"e2e_tests": [], "edge_cases": [], "business_tests": [{
                "id": "BUS-E2E-1",
                "title": "Unauthenticated customer cannot create an order",
                "intent": "Only authenticated customers may create orders.",
                "steps": [],
            }]}
        return super().chat_json(system, user, **kwargs)


def _run(store, llm):
    generate_fresh_testcases_pipeline(store, llm, FakeEmbedder(), {
        "feature_id": "feature-1",
        "total": 10,
        "focus": {"functional": 20, "e2e": 20, "api": 20, "nfr": 20, "ui": 20},
    })
    calls = llm.calls_matching(FALLBACK_MARKER)
    assert len(calls) == 1, "expected exactly one business fallback call"
    return calls[0]["user"]


def test_fallback_asks_only_for_the_shortfall_and_lists_covered_rules():
    # 5 rules -> floor min(10, max(3, round(5 * 0.6))) = 3. The E2E call wrote 1, so the
    # fallback may add at most 2, and it is told which rule is already covered.
    prompt = _run(FiveRuleStore(), OneBusinessTestLLM())
    assert "Generate at most 2 more tests" in prompt
    assert "Output at most 2 tests" in prompt
    assert "ALREADY COVERED" in prompt
    assert "- Only authenticated customers may create orders." in prompt
    assert "Generate exactly 5 tests" not in prompt


def test_fallback_with_no_existing_tests_asks_for_the_floor_not_every_rule():
    # The base FakeLLM's E2E call returns no business tests: the floor is 3, so the
    # fallback asks for at most 3 tests instead of one for each of the 5 rules.
    prompt = _run(FiveRuleStore(), SpyLLM())
    assert "Generate at most 3 tests, one per distinct rule" in prompt
    assert "Output at most 3 tests" in prompt
    assert "ALREADY COVERED" not in prompt
    assert "Generate exactly 5 tests" not in prompt


def test_prompt_without_top_up_arguments_is_unchanged():
    context = {"featureName": "Create order", "businessContext": {"requirements": RULES}}
    prompt = build_business_test_prompt(context, {}, top_k=4)
    assert "Generate exactly 5 tests — one per rule." in prompt
    assert "Output at most 5 tests" in prompt
    assert "ALREADY COVERED" not in prompt


def test_top_up_never_asks_for_more_than_the_rule_count():
    context = {"featureName": "Create order", "businessContext": {"requirements": RULES[:2]}}
    prompt = build_business_test_prompt(context, {}, top_k=4, existing_tests=[], max_new=3)
    assert "Generate at most 2 tests" in prompt
    assert "Output at most 2 tests" in prompt
