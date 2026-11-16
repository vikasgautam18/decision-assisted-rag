from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import Field

from pipelines.contracts import Artifact, Identifier, Text


class Operand(Artifact):
    source_id: Identifier
    quote: Text
    value: str = Field(pattern=r"^-?\d+(?:\.\d+)?$", max_length=40)
    unit: str = Field(min_length=1, max_length=50)


class ComputedOperand(Artifact):
    calculation_id: Identifier
    value: str = Field(pattern=r"^-?\d+(?:\.\d+)?$", max_length=40)
    unit: str = Field(min_length=1, max_length=50)


class LineOperand(Artifact):
    source_id: Identifier
    line_numbers: tuple[Annotated[int, Field(ge=1, le=10000)], ...] = Field(
        min_length=1, max_length=32
    )
    value: str = Field(pattern=r"^-?\d+(?:\.\d+)?$", max_length=40)
    unit: str = Field(min_length=1, max_length=50)


class Calculation(Artifact):
    calculation_id: Identifier
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
        "percent_change",
        "mean",
        "apply_percent",
        "reduce_percent",
        "percent_of",
        "ratio_percent",
        "f_to_c",
    ]
    operands: tuple[Operand | ComputedOperand | LineOperand, ...] = Field(
        min_length=1, max_length=12
    )
    result: str = Field(pattern=r"^-?\d+(?:\.\d+)?$", max_length=40)
    result_unit: str = Field(min_length=1, max_length=50)
    decimal_places: int = Field(default=2, ge=0, le=8)
    rounding: Literal["half_even", "half_up"] = "half_even"
    claim_id: Identifier


class CountItem(Artifact):
    source_id: Identifier
    quote: Text
    item_key: str = Field(min_length=1, max_length=100)


class CountCalculation(Artifact):
    calculation_id: Identifier
    operation: Literal["count"]
    items: tuple[CountItem, ...] = Field(min_length=1, max_length=100)
    result: str = Field(pattern=r"^\d+$", max_length=4)
    result_unit: Literal["scalar"] = "scalar"
    claim_id: Identifier


CalculationRecord = Calculation | CountCalculation


def has_number(text: str, expected: str) -> bool:
    return any(
        Decimal(number.replace(",", "")) == Decimal(expected)
        for number in re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])", text)
    )


def content_without_coordinates(text: str) -> str:
    return re.sub(
        r"(?m)^(?:row \d+, column \d+, kind \w+, span \d+x\d+: |\[page \d+\] )",
        "",
        text,
    )


def exact_ordered_excerpt(quote: str, source: str) -> bool:
    position = 0
    for line in quote.splitlines():
        if not line:
            continue
        found = source.find(line, position)
        if found < 0:
            return False
        position = found + len(line)
    return position > 0


def verify_calculations(
    calculations: tuple[CalculationRecord, ...], sources: list[dict], draft: str, claims: tuple
) -> None:
    if len({item.calculation_id for item in calculations}) != len(calculations):
        raise ValueError("duplicate calculation ID")
    evidence = {source["id"]: source["text"] for source in sources}
    documents = {source["id"]: source.get("document", source["id"]) for source in sources}
    claim_map = {claim.claim_id: claim for claim in claims}
    computed: dict[str, tuple[Decimal, str, set[str]]] = {}
    for calculation in calculations:
        claim = claim_map.get(calculation.claim_id)
        if claim is None:
            raise ValueError("calculation must map to a material claim")
        used_sources: set[str] = set()
        if isinstance(calculation, CountCalculation):
            keys = set()
            for item in calculation.items:
                if (
                    item.source_id not in evidence
                    or item.source_id not in claim.source_ids
                    or not exact_ordered_excerpt(item.quote, evidence[item.source_id])
                    or not re.search(
                        rf"(?<![\w-]){re.escape(item.item_key)}(?![\w-])",
                        content_without_coordinates(item.quote),
                    )
                ):
                    raise ValueError("count item lacks exact cited provenance or identity")
                key = (documents[item.source_id], item.item_key)
                if key in keys:
                    raise ValueError("count duplicates a documentary item")
                keys.add(key)
                used_sources.add(item.source_id)
            if Decimal(calculation.result) != len(keys):
                raise ValueError("count result does not match distinct cited items")
            if not has_number(draft, calculation.result) or not has_number(
                claim.text, calculation.result
            ):
                raise ValueError("count result must appear in both draft and mapped claim")
            computed[calculation.calculation_id] = (
                Decimal(calculation.result),
                calculation.result_unit,
                used_sources,
            )
            continue
        values = []
        for operand in calculation.operands:
            if isinstance(operand, ComputedOperand):
                previous = computed.get(operand.calculation_id)
                if previous is None:
                    raise ValueError("computed operand must reference an earlier calculation")
                value, unit, source_ids = previous
                if (
                    value != Decimal(operand.value)
                    or unit != operand.unit
                    or not source_ids <= set(claim.source_ids)
                ):
                    raise ValueError(
                        "computed operand lacks matching value, unit or transitive citations"
                    )
                values.append(value)
                used_sources.update(source_ids)
                continue
            quote = operand.quote if isinstance(operand, Operand) else ""
            if isinstance(operand, LineOperand):
                if operand.source_id not in evidence:
                    raise ValueError("line operand has an unknown source")
                lines = evidence[operand.source_id].splitlines()
                if tuple(
                    sorted(set(operand.line_numbers))
                ) != operand.line_numbers or operand.line_numbers[-1] > len(lines):
                    raise ValueError("line operand requires ordered unique existing source lines")
                quote = "\n".join(lines[number - 1] for number in operand.line_numbers)
            if (
                operand.source_id not in evidence
                or operand.source_id not in claim.source_ids
                or not exact_ordered_excerpt(quote, evidence[operand.source_id])
            ):
                raise ValueError(
                    f"calculation input lacks exact cited provenance: {operand.source_id}"
                )
            documentary_text = content_without_coordinates(quote)
            if not has_number(documentary_text, operand.value):
                raise ValueError("calculation input value is absent from its source quote")
            percent_words = operand.unit == "%" and re.search(
                r"\b(?:per\s+cent|percent)\b", documentary_text, re.IGNORECASE
            )
            if (
                operand.unit != "scalar"
                and operand.unit.casefold() not in documentary_text.casefold()
                and not percent_words
            ):
                raise ValueError("calculation input unit is absent from its source quote")
            values.append(Decimal(operand.value))
            used_sources.add(operand.source_id)
        units = [operand.unit for operand in calculation.operands]
        operation = calculation.operation
        if operation == "f_to_c":
            if len(values) != 1 or units[0] not in ("F", "°F", "ºF"):
                raise ValueError("Fahrenheit conversion requires one Fahrenheit operand")
            if calculation.result_unit not in ("C", "°C", "ºC"):
                raise ValueError("Fahrenheit conversion requires a Celsius result")
            expected_unit = calculation.result_unit
        elif operation in ("apply_percent", "reduce_percent", "percent_of"):
            if len(values) != 2 or units[1] != "%":
                raise ValueError("percentage application requires a base and sourced percentage")
            if operation == "reduce_percent" and values[1] < 0:
                raise ValueError("reduce_percent requires a nonnegative reduction magnitude")
            expected_unit = units[0]
        elif operation in ("add", "subtract", "mean", "percent_change", "ratio_percent"):
            if len(set(units)) != 1:
                raise ValueError("calculation requires matching units")
            expected_unit = "%" if operation in ("percent_change", "ratio_percent") else units[0]
        elif operation == "multiply":
            dimensional = [unit for unit in units if unit != "scalar"]
            if len(dimensional) > 1:
                raise ValueError("multiplication permits only one dimensional operand")
            expected_unit = dimensional[0] if dimensional else "scalar"
        else:
            expected_unit = "scalar" if units[0] == units[1] else f"{units[0]}/{units[1]}"
        if calculation.result_unit != expected_unit:
            raise ValueError("calculation result unit does not match operation")
        if operation != "f_to_c" and len(values) < 2:
            raise ValueError("arithmetic requires at least two operands")
        if (
            operation in ("subtract", "divide", "percent_change", "ratio_percent")
            and len(values) != 2
        ):
            raise ValueError("binary calculation requires two operands")
        try:
            if operation == "add":
                result = sum(values)
            elif operation == "subtract":
                result = values[0] - values[1]
            elif operation == "multiply":
                result = Decimal(1)
                for value in values:
                    result *= value
            elif operation == "divide":
                result = values[0] / values[1]
            elif operation == "percent_change":
                result = (values[1] - values[0]) / values[0] * 100
            elif operation == "ratio_percent":
                result = values[0] / values[1] * 100
            elif operation == "apply_percent":
                result = values[0] * (1 + values[1] / 100)
            elif operation == "reduce_percent":
                result = values[0] * (1 - values[1] / 100)
            elif operation == "percent_of":
                result = values[0] * values[1] / 100
            elif operation == "f_to_c":
                result = (values[0] - 32) * 5 / 9
            else:
                result = sum(values) / len(values)
            expected = result.quantize(
                Decimal(1).scaleb(-calculation.decimal_places),
                rounding=ROUND_HALF_UP if calculation.rounding == "half_up" else ROUND_HALF_EVEN,
            )
        except (InvalidOperation, ZeroDivisionError) as exc:
            raise ValueError("invalid or undefined calculation") from exc
        if Decimal(calculation.result) != expected:
            raise ValueError("calculated result does not match deterministic arithmetic")
        if not has_number(draft, calculation.result) or not has_number(
            claim.text, calculation.result
        ):
            raise ValueError("calculated result must appear in both draft and mapped claim")
        computed[calculation.calculation_id] = (
            Decimal(calculation.result),
            calculation.result_unit,
            used_sources,
        )
