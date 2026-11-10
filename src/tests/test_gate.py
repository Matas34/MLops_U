from check_quality_gate import check_quality_gate


def test_gate_passes_when_score_is_higher():
    assert check_quality_gate(r2=0.91, threshold=0.9) is True

def test_gate_passes_when_score_is_exactly_equal():
    assert check_quality_gate(r2=0.90, threshold=0.9) is True

def test_gate_fails_when_score_is_lower():
    assert check_quality_gate(r2=0.89, threshold=0.9) is False