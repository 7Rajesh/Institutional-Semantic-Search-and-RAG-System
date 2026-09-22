from isearch.verify import check_answer, unsupported_numbers

EVIDENCE = ("A student may withdraw until the end of week 8. A student must keep at least 12 credits. "
           "Fee is Rs 1,25,000 per semester. Gate closes at 22:30 for residents. CGPA below 5.0. "
           "Attend at least 75% of the classes.")


def test_accepts_correct_numbers_in_context():
    ok = [
        "Gate closes at 22:30 [1].",
        "The fee is Rs 125000 per semester [1].",
        "Withdraw until the end of week 8 [1].",
        "1. Withdraw by week 8 [1].\n2. CGPA below 5.0 [1].",
    ]
    for a in ok:
        assert check_answer(a, [EVIDENCE]) == [], a


def test_rejects_number_from_a_different_context():
    # "12" appears in the evidence (12 credits), but not next to "week" - this must be rejected.
    problems = check_answer("Withdraw until the end of week 12 [1].", [EVIDENCE])
    assert problems and "numbers not found" in problems[0]


def test_rejects_wrong_number_entirely():
    assert check_answer("Gate closes at 23:30 [1].", [EVIDENCE])


def test_rejects_missing_or_invalid_citations():
    assert "no citations" in check_answer("Gate closes at 22:30.", [EVIDENCE])[0]
    assert "do not exist" in check_answer("Gate closes at 22:30 [4].", [EVIDENCE])[0]


def test_unsupported_numbers_is_comma_insensitive():
    assert unsupported_numbers("Fee is 125000 [1]", "Fee is 1,25,000") == []
