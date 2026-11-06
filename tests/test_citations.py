from dronedocs.citations import check_citations, extract_citations, is_refusal, strip_invalid_citations

CTX = {"drone_rules_2021.pdf#c003", "px4.md#c010"}


def test_extracts_single_and_grouped_citations():
    txt = "Nano is <=250 g [drone_rules_2021.pdf#c003]. Disarm is 2 s [px4.md#c010, drone_rules_2021.pdf#c003]."
    assert extract_citations(txt) == ["drone_rules_2021.pdf#c003", "px4.md#c010"]


def test_valid_answer_passes():
    chk = check_citations("Nano drones weigh at most 250 grams [drone_rules_2021.pdf#c003].", CTX)
    assert chk.status == "ok" and chk.valid_ids == ["drone_rules_2021.pdf#c003"] and chk.coverage == 1.0


def test_invented_citation_ids_are_rejected():
    chk = check_citations("Penalty is one lakh rupees [drone_rules_2021.pdf#c999].", CTX)
    assert chk.status == "refused_no_citations"
    assert chk.invalid_ids == ["drone_rules_2021.pdf#c999"]


def test_known_but_not_in_context_is_invalid():
    chk = check_citations("Something true [other.md#c001].", CTX, known_ids=CTX | {"other.md#c001"})
    assert chk.refused and chk.invalid_ids == ["other.md#c001"]


def test_uncited_answer_is_refused():
    chk = check_citations("Drones under 250 g are nano. They need no licence. No insurance needed either.", CTX)
    assert chk.status == "refused_no_citations"


def test_mostly_uncited_answer_is_refused():
    ans = (
        "Nano is 250 grams [px4.md#c010]. The licence lasts ten years always. Fees are one hundred rupees. "
        "Pilots must be adults of any age."
    )
    assert check_citations(ans, CTX, max_uncited_sentence_ratio=0.5).status == "refused_uncited"


def test_model_refusal_sentinel():
    assert check_citations("INSUFFICIENT_EVIDENCE", CTX).status == "refused_by_model"
    assert is_refusal("") and is_refusal("  INSUFFICIENT_EVIDENCE  ")


def test_strip_invalid():
    assert strip_invalid_citations("A [x#c001] b [y#c002].", ["x#c001"]) == "A b [y#c002]."


def test_gpt_oss_lenticular_brackets_are_normalized():
    chk = check_citations("Nano is at most 250 grams【drone_rules_2021.pdf#c003】.", CTX)
    assert chk.status == "ok" and chk.valid_ids == ["drone_rules_2021.pdf#c003"]
