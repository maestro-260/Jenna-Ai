import pytest
from services.security_service import SecurityMonitor


@pytest.mark.parametrize("text,expected", [
    ("sudo rm -rf /", True),
    ("Please help me with math", False),
    ("My SSN is 123-45-6789", True),
    ("What's the weather today?", False)
])
def test_threat_detection(text, expected):
    monitor = SecurityMonitor()
    result = monitor.analyze(text)
    actual_threat = (
        result['threat_score'] > 0.8 or len(result['pattern_matches']) > 0
    )
    assert actual_threat == expected, f"Failed for: '{text}'"
