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


def test_expected_answer_type_numeric_score():
    assert expected_answer_type("what credit score do you need for the card") == "numeric"


def test_expected_answer_type_numeric_weight():
    assert expected_answer_type("what was marvin hagler's fighting weight") == "numeric"


def test_expected_answer_type_numeric_population():
    assert expected_answer_type("what is the population of tokyo") == "numeric"


def test_expected_answer_type_numeric_hindi_credit_score():
    assert expected_answer_type("एक्सपीडिया कार्ड के लिए आपको किस तरह का क्रेडिट स्कोर मंजूरी दी जानी चाहिए") == "numeric"


def test_expected_answer_type_numeric_hindi_kitna():
    assert expected_answer_type("ओक्लाहोमा में सबसे अधिक तापमान कितना दर्ज किया गया") == "numeric"


def test_expected_answer_type_numeric_hindi_kab():
    assert expected_answer_type("कब भारत को आजादी मिली") == "numeric"


def test_expected_answer_type_person_hindi_kaun():
    assert expected_answer_type("कौन भारत का प्रधानमंत्री है") == "person"


def test_expected_answer_type_numeric_tamil_evvalavu():
    assert expected_answer_type("இந்த பொருளின் விலை எவ்வளவு") == "numeric"


def test_expected_answer_type_numeric_tamil_edai():
    assert expected_answer_type("அவரின் எடை என்ன") == "numeric"


def test_expected_answer_type_person_tamil_yaar():
    assert expected_answer_type("யார் இந்தியாவின் பிரதமர்") == "person"


def test_expected_answer_type_hindi_kya_not_forced_yesno():
    # "क्या" prefixes both yes/no questions and "what is" description
    # queries in Hindi — deliberately NOT classified, since guessing wrong
    # would gate a valid answer on the wrong check.
    assert expected_answer_type("क्या गैस प्रोग्राम है") is None


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


def test_span_satisfies_expected_type_numeric_devanagari_digits():
    assert span_satisfies_expected_type("numeric", "मार्टिन लूथर किंग की मृत्यु ५१ वर्ष की आयु में हुई।") is True


def test_span_satisfies_expected_type_person_hindi_span_not_blocked():
    # Devanagari has no letter-casing, so the capitalization heuristic can't
    # verify this — it must not treat that as a negative signal.
    assert span_satisfies_expected_type("person", "प्रधानमंत्री नरेंद्र मोदी हैं।") is True
