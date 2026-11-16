import asyncio
import json

import pytest

from clients.decision import ChoiceResult
from clients.ordinary_decisions import OrdinaryDecisions
from pipelines.budget import ServiceError

QUESTIONS = {
    "support": (
        "Check whether the evidence supports the claim.",
        {"supported": "supported", "contradicted": "contradicted", "insufficient": "unclear"},
    ),
    "fitment": ("Check fitment.", {"yes": "fitted", "no": "not fitted"}),
}


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def semantic_complete(self, system, content):
        self.calls.append((system, content))
        return self.response


def test_ordinary_choices_map_to_choice_results_without_probabilities():
    models = FakeModels(json.dumps({"answers": {"support": "supported", "fitment": "no"}}))
    result = asyncio.run(OrdinaryDecisions(models).decide({"text": "source"}, QUESTIONS))
    assert result == {
        "support": ChoiceResult("supported", None),
        "fitment": ChoiceResult("no", None),
    }
    system, content = models.calls[0]
    request = json.loads(content)
    assert "Do not include probabilities" in system
    assert request["questions"]["support"]["allowed_labels"] == [
        "supported",
        "contradicted",
        "insufficient",
    ]
    assert request["questions"]["support"]["criteria"] == QUESTIONS["support"][1]


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        json.dumps({"answers": {"support": "supported"}}),
        json.dumps({"answers": {"support": "supported", "fitment": "no", "extra": "x"}}),
        json.dumps({"answers": {"support": "maybe", "fitment": "no"}}),
        json.dumps({"answers": {"support": "supported", "fitment": 1}}),
        json.dumps({"answers": {"support": "supported", "fitment": "no"}, "extra": True}),
    ],
)
def test_malformed_or_non_categorical_responses_are_explicit_failures(response):
    models = FakeModels(response)
    with pytest.raises(ServiceError):
        asyncio.run(OrdinaryDecisions(models).decide({}, QUESTIONS))
    assert len(models.calls) == 1
