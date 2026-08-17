from speculative_rag.answer_type import expected_answer_type, span_satisfies_expected_type


def test_expected_answer_type_yesno():
    assert expected_answer_type("is it true a summons is a call to appear in court") == "yesno"


def test_expected_answer_type_numeric_what_year():
    assert expected_answer_type("what year did facebook ipo") == "numeric"


def test_expected_answer_type_numeric_how_many():
    assert expected_answer_type("how many calories in an egg") == "numeric"


def test_expected_answer_type_numeric_when():
    assert expected_answer_type("when did world war 2 end") == "numeric"


def test_expected_answer_type_person():
    assert expected_answer_type("who is the president of the united states") == "person"


def test_expected_answer_type_unclassified_description_query():
    assert expected_answer_type("kids definition of multiples") is None


def test_span_satisfies_expected_type_none_always_true():
    assert span_satisfies_expected_type(None, "") is True
    assert span_satisfies_expected_type(None, "the second quarter") is True


def test_span_satisfies_expected_type_yesno_positive():
    assert span_satisfies_expected_type("yesno", "Yes, that is correct.") is True


def test_span_satisfies_expected_type_yesno_negative():
    assert span_satisfies_expected_type("yesno", "A summons is a legal document.") is False


def test_span_satisfies_expected_type_numeric_positive():
    assert span_satisfies_expected_type("numeric", "Facebook IPO'd in 2012.") is True


def test_span_satisfies_expected_type_numeric_negative():
    assert span_satisfies_expected_type("numeric", "It happened in the second quarter.") is False


def test_span_satisfies_expected_type_person_positive():
    assert span_satisfies_expected_type("person", "The president is Joe Biden.") is True


def test_span_satisfies_expected_type_person_negative():
    assert span_satisfies_expected_type("person", "the current officeholder took office recently.") is False
