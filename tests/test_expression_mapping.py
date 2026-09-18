"""Tests for the temporary software-baseline expression mapper."""

import pytest

import inmoove.face as face
from inmoove.face import FaceExpression, MockHead


NEUTRAL = (0.5,) * 16
HAPPY_3 = (
    0.5,
    0.6,
    0.5,
    0.6,
    0.85,
    0.15,
    0.85,
    0.15,
    0.6,
    0.6,
    0.75,
    0.75,
    0.4,
    0.4,
    0.7,
    0.3,
)
SAD_3 = (
    0.5,
    0.3,
    0.5,
    0.3,
    0.6,
    0.4,
    0.6,
    0.4,
    0.3,
    0.3,
    0.2,
    0.2,
    0.7,
    0.7,
    0.3,
    0.2,
)


def _map(expression: str, intensity: int) -> FaceExpression:
    mapper = getattr(face, "map_expression", None)
    assert callable(mapper), "inmoove.face.map_expression must be public"
    return mapper(expression, intensity)


def _interpolate(target: tuple[float, ...], intensity: int) -> tuple[float, ...]:
    return tuple(0.5 + (value - 0.5) * intensity / 3 for value in target)


@pytest.mark.parametrize(
    ("expression", "intensity", "expected"),
    [
        ("neutral", 1, NEUTRAL),
        ("neutral", 2, NEUTRAL),
        ("neutral", 3, NEUTRAL),
        ("happy", 1, _interpolate(HAPPY_3, 1)),
        ("happy", 2, _interpolate(HAPPY_3, 2)),
        ("happy", 3, HAPPY_3),
        ("sad", 1, _interpolate(SAD_3, 1)),
        ("sad", 2, _interpolate(SAD_3, 2)),
        ("sad", 3, SAD_3),
    ],
)
def test_map_expression_supports_all_baseline_combinations(
    expression, intensity, expected
):
    result = _map(expression, intensity)

    values = result.get_all_values()
    assert isinstance(result, FaceExpression)
    assert len(values) == 16
    assert values == pytest.approx(expected)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_map_expression_is_deterministic_for_happy_intensity_two():
    first = _map("happy", 2).get_all_values()
    second = _map("happy", 2).get_all_values()

    assert first == second
    assert len(first) == 16


@pytest.mark.parametrize(
    ("expression", "intensity"),
    [
        ("happy", 2),
        ("neutral", 1),
    ],
)
def test_mapped_expression_flows_into_mock_head(expression, intensity):
    """A Step 3 label reaches MockHead as the same complete 16D expression."""
    generated = _map(expression, intensity)
    head = MockHead(verbose=False)

    head.set_expression(generated)

    stored = head.get_current_expression()
    history = head.get_expression_history()
    assert stored is not None
    assert stored.get_all_values() == generated.get_all_values()
    assert len(stored.get_all_values()) == 16
    assert len(history) == 1
    assert history[0]["expression"].get_all_values() == generated.get_all_values()


@pytest.mark.parametrize("expression", ["angry", "Happy", ""])
def test_map_expression_rejects_unsupported_expression(expression):
    with pytest.raises(ValueError, match="unsupported expression"):
        _map(expression, 2)


@pytest.mark.parametrize("intensity", [0, 4, 1.5, "2", True])
def test_map_expression_rejects_invalid_intensity(intensity):
    with pytest.raises(ValueError, match="intensity"):
        _map("happy", intensity)
