import pytest

from pipelines.calculations import (
    Calculation,
    ComputedOperand,
    CountCalculation,
    CountItem,
    LineOperand,
    Operand,
    exact_ordered_excerpt,
    has_number,
    verify_calculations,
)
from pipelines.contracts import MaterialClaim

SOURCE = {"id": "test-table", "text": "Current 310 km; previous 300 km; zero 0 km"}
CLAIM = MaterialClaim(claim_id="c1", text="Difference 10 km", source_ids=("test-table",))


def calculation(**changes):
    return Calculation(
        **{
            "calculation_id": "calc1",
            "operation": "subtract",
            "operands": (
                Operand(source_id="test-table", quote="Current 310 km", value="310", unit="km"),
                Operand(source_id="test-table", quote="previous 300 km", value="300", unit="km"),
            ),
            "result": "10",
            "result_unit": "km",
            "claim_id": "c1",
            **changes,
        }
    )


def test_source_bound_arithmetic_accepts_exact_result():
    verify_calculations((calculation(),), [SOURCE], "Difference 10 km", (CLAIM,))


def test_table_excerpts_may_omit_lines_but_never_invent_or_reorder_them():
    source = "column 1: Capacity (kWh)\ncolumn 2: Cost\nrow 3: 58.0\nrow 4: 44.0"
    assert exact_ordered_excerpt("column 1: Capacity (kWh)\nrow 3: 58.0", source)
    assert not exact_ordered_excerpt("row 3: 58.0\ncolumn 1: Capacity (kWh)", source)
    assert not exact_ordered_excerpt("column 1: Capacity (kWh)\nrow 3: 59.0", source)


@pytest.mark.parametrize(
    "changes",
    [
        {"result": "11"},
        {"result_unit": "m"},
        {"claim_id": "foreign"},
        {
            "operands": (
                Operand(source_id="test-table", quote="Current 310 km", value="999", unit="km"),
                Operand(source_id="test-table", quote="previous 300 km", value="300", unit="km"),
            )
        },
        {
            "operands": (
                Operand(source_id="test-table", quote="invented 310 km", value="310", unit="km"),
                Operand(source_id="test-table", quote="previous 300 km", value="300", unit="km"),
            )
        },
    ],
)
def test_invalid_math_or_provenance_never_passes(changes):
    with pytest.raises(ValueError):
        verify_calculations((calculation(**changes),), [SOURCE], "Difference 10 km", (CLAIM,))


def test_zero_division_is_explicit_invalid_output():
    record = calculation(
        operation="divide",
        result_unit="scalar",
        operands=(
            Operand(source_id="test-table", quote="Current 310 km", value="310", unit="km"),
            Operand(source_id="test-table", quote="zero 0 km", value="0", unit="km"),
        ),
    )
    with pytest.raises(ValueError, match="undefined"):
        verify_calculations((record,), [SOURCE], "10", (CLAIM,))


def test_mixed_unit_addition_rejected():
    record = calculation(
        operands=(
            Operand(source_id="test-table", quote="Current 310 km", value="310", unit="km"),
            Operand(source_id="test-table", quote="previous 300 km", value="300", unit="scalar"),
        )
    )
    with pytest.raises(ValueError, match="matching units"):
        verify_calculations((record,), [SOURCE], "10", (CLAIM,))


def test_percentage_change_is_rounded_deterministically():
    record = calculation(
        operation="percent_change",
        result="3.33",
        result_unit="%",
        operands=(
            Operand(source_id="test-table", quote="previous 300 km", value="300", unit="km"),
            Operand(source_id="test-table", quote="Current 310 km", value="310", unit="km"),
        ),
    )
    claim = CLAIM.model_copy(update={"text": "Increase 3.33%"})
    verify_calculations((record,), [SOURCE], "Increase 3.33%", (claim,))


@pytest.mark.parametrize(
    ("operation", "rate", "result"),
    [
        ("apply_percent", "-6", "280.12"),
        ("reduce_percent", "6", "280.12"),
        ("percent_of", "-6", "-17.88"),
    ],
)
def test_explicit_percentage_operations_use_sourced_rates(operation, rate, result):
    source = {"id": "table", "text": f"Declared range 298 km\nCorrection {rate} per cent"}
    record = calculation(
        operation=operation,
        operands=(
            Operand(source_id="table", quote="Declared range 298 km", value="298", unit="km"),
            Operand(source_id="table", quote=f"Correction {rate} per cent", value=rate, unit="%"),
        ),
        result=result,
    )
    claim = MaterialClaim(claim_id="c1", text=f"Result {result} km", source_ids=("table",))
    verify_calculations((record,), [source], claim.text, (claim,))
    bad = record.model_copy(
        update={
            "operands": (
                record.operands[0],
                record.operands[1].model_copy(update={"value": "0.06"}),
            )
        }
    )
    with pytest.raises(ValueError, match="absent"):
        verify_calculations((bad,), [source], claim.text, (claim,))


def test_ratio_percentage_has_explicit_rounding_and_numeric_text_matching():
    source = {"id": "table", "text": "Usable 14.5 kWh; gross 16.0 kWh"}
    operands = (
        Operand(source_id="table", quote=source["text"], value="14.5", unit="kWh"),
        Operand(source_id="table", quote=source["text"], value="16.0", unit="kWh"),
    )
    for rounding, result in (("half_even", "90.62"), ("half_up", "90.63")):
        record = calculation(
            operation="ratio_percent",
            operands=operands,
            result=result,
            result_unit="%",
            rounding=rounding,
        )
        claim = MaterialClaim(claim_id="c1", text=f"Share {result}%", source_ids=("table",))
        verify_calculations((record,), [source], claim.text, (claim,))
    assert has_number("Difference 1,310.0 km", "1310.00")
    assert not has_number("A value 1310", "310")


def test_authorized_fahrenheit_conversion_is_exact_and_unit_bound():
    source = {"id": "table", "text": "DPF continuous limit 617 ºF"}
    record = calculation(
        operation="f_to_c",
        operands=(Operand(source_id="table", quote=source["text"], value="617", unit="ºF"),),
        result="325",
        result_unit="°C",
        decimal_places=0,
    )
    claim = MaterialClaim(claim_id="c1", text="Limit 325 °C", source_ids=("table",))
    verify_calculations((record,), [source], claim.text, (claim,))
    with pytest.raises(ValueError, match="Celsius"):
        verify_calculations(
            (record.model_copy(update={"result_unit": "K"}),), [source], claim.text, (claim,)
        )


def test_count_uses_distinct_source_identity_not_invented_numeric_operands():
    sources = [
        {"id": "page1", "document": "test", "text": "TEST-001 passed\nTEST-002 passed"},
        {"id": "page2", "document": "test", "text": "TEST-001 passed"},
    ]
    record = CountCalculation(
        calculation_id="count",
        operation="count",
        result="2",
        claim_id="c1",
        items=(
            CountItem(source_id="page1", quote="TEST-001 passed", item_key="TEST-001"),
            CountItem(source_id="page1", quote="TEST-002 passed", item_key="TEST-002"),
        ),
    )
    claim = MaterialClaim(claim_id="c1", text="2 tests", source_ids=("page1", "page2"))
    verify_calculations((record,), sources, claim.text, (claim,))
    duplicate = record.model_copy(
        update={
            "items": (
                record.items[0],
                CountItem(source_id="page2", quote="TEST-001 passed", item_key="TEST-001"),
            )
        }
    )
    with pytest.raises(ValueError, match="duplicates"):
        verify_calculations((duplicate,), sources, claim.text, (claim,))


def test_chained_calculation_requires_verified_prior_value_and_transitive_citations():
    first = calculation()
    second = calculation(
        calculation_id="calc2",
        claim_id="c2",
        operation="subtract",
        operands=(
            ComputedOperand(calculation_id="calc1", value="10", unit="km"),
            Operand(source_id="test-table", quote="zero 0 km", value="0", unit="km"),
        ),
        result="10",
    )
    second_claim = CLAIM.model_copy(update={"claim_id": "c2"})
    verify_calculations((first, second), [SOURCE], CLAIM.text, (CLAIM, second_claim))
    for changed in (
        {"calculation_id": "future"},
        {"value": "11"},
        {"unit": "m"},
    ):
        bad = second.model_copy(
            update={
                "operands": (
                    second.operands[0].model_copy(update=changed),
                    second.operands[1],
                )
            }
        )
        with pytest.raises(ValueError):
            verify_calculations((first, bad), [SOURCE], CLAIM.text, (CLAIM, second_claim))


def test_source_line_addresses_preserve_unusual_temperature_glyphs_without_fuzzy_quotes():
    source = {
        "id": "table",
        "text": "row 0, column 0, kind columnHeader, span 1x1: Range (km)\n"
        "row 1, column 0, kind content, span 1x1: 72\n"
        "row 4, column 0, kind content, span 1x1: Above 40 °℃ up to 48 ℃\n"
        "row 4, column 1, kind content, span 1x1: -10 %",
    }
    record = calculation(
        operation="apply_percent",
        operands=(
            LineOperand(source_id="table", line_numbers=(1, 2), value="72", unit="km"),
            LineOperand(source_id="table", line_numbers=(3, 4), value="-10", unit="%"),
        ),
        result="64.8",
        decimal_places=1,
    )
    claim = MaterialClaim(claim_id="c1", text="Range 64.8 km", source_ids=("table",))
    verify_calculations((record,), [source], claim.text, (claim,))
    for lines in ((4, 3), (3, 3), (3, 5)):
        bad = record.model_copy(
            update={
                "operands": (
                    record.operands[0],
                    record.operands[1].model_copy(update={"line_numbers": lines}),
                )
            }
        )
        with pytest.raises(ValueError, match="ordered unique existing"):
            verify_calculations((bad,), [source], claim.text, (claim,))


def test_coordinate_metadata_is_not_a_documentary_numeric_operand():
    source = {"id": "table", "text": "row 1, column 0, kind content, span 1x1: 72 km"}
    record = calculation(
        operation="f_to_c",
        operands=(LineOperand(source_id="table", line_numbers=(1,), value="1", unit="scalar"),),
        result="1",
    )
    claim = MaterialClaim(claim_id="c1", text="1", source_ids=("table",))
    with pytest.raises(ValueError, match="absent"):
        verify_calculations((record,), [source], claim.text, (claim,))
