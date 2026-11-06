from dronedocs.citations import REFUSAL_TEXT


class Recorder:
    def __init__(self):
        self.stages = []
        self.results = []

    def on_stage(self, rid, stage, payload, dt):
        self.stages.append(stage)

    def on_result(self, res):
        self.results.append(res)


def test_happy_path_returns_cited_answer(make_pipeline):
    rec = Recorder()
    pipe = make_pipeline("A nano drone weighs at most 250 grams [drone_rules_2021.pdf#c000].", observers=[rec])
    res = pipe.ask("max weight of nano drone?")
    assert not res.refused
    assert res.citations[0].chunk_id == "drone_rules_2021.pdf#c000"
    assert res.citations[0].locator == "page 27"
    assert rec.stages == ["retrieve", "rerank", "evidence_gate", "generate", "validate_citations"]
    assert rec.results[0].request_id == res.request_id
    assert res.prompt_version.startswith("answer-v")
    # the prompt actually contains the chunk ids the model is allowed to cite
    assert "drone_rules_2021.pdf#c000" in res.prompt_messages[1]["content"]


def test_evidence_gate_refuses_without_calling_llm(make_pipeline):
    pipe = make_pipeline("should never be produced", rerank_score=-9.0)
    res = pipe.ask("what is the FAA part 107 fee?")
    assert res.refused and res.refusal_reason == "refused_by_gate" and res.answer == REFUSAL_TEXT
    assert pipe.llm.calls == []


def test_hallucinated_citation_turns_into_refusal(make_pipeline):
    res = make_pipeline("The fee is Rs 5000 [drone_rules_2021.pdf#c777].").ask("fee?")
    assert res.refused and res.refusal_reason == "refused_no_citations" and res.citations == []


def test_model_refusal_propagates(make_pipeline):
    res = make_pipeline("INSUFFICIENT_EVIDENCE").ask("mavic 3 weight?")
    assert res.refused and res.refusal_reason == "refused_by_model"


def test_observer_failure_does_not_break_serving(make_pipeline):
    class Boom:
        def on_stage(self, *a):
            raise RuntimeError("tracer down")

    res = make_pipeline("Nano is 250 g [drone_rules_2021.pdf#c000].", observers=[Boom()]).ask("nano?")
    assert not res.refused
