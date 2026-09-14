"""Classify with AI must only ever read free-text/category columns: a real run
picked the image ``filename`` column for the goal "Cu in the file" and built a
meaningless "does the filename contain a decimal?" layer. ID, file-name, date,
coordinate/depth and numeric columns are never offered to the AI, and the chat
tools refuse them when passed directly."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from direct.classify_text_to_voxel import _candidate_columns, pick_text_column


def _resolvesa_like() -> pd.DataFrame:
    return pd.DataFrame({
        "filename": ["9955_5_9_9.15.jpg", "9955_5_8_9.04.jpg", "311165_3_7_108.66.jpg"],
        "drillhole_number": [9955, 9955, 311165],
        "core": [5, 5, 3],
        "LOG_NUMBER": [101, 101, 202],
        "LOGGING_ORGANISATION": ["GSSA", "GSSA", "Minotaur"],
        "LOGGING_DATE": ["2019-01-02", "2019-01-02", "2020-05-06"],
        "depth_from_m": [9.0, 8.9, 100.0],
        "depth_to_m": [9.3, 9.1, 110.0],
        "major_lithology": ["Granite", "Granite", "Gneiss"],
        "DESCRIPTION": ["pink granite with pyrite", "granite, weak chlorite", "augen gneiss, chip sample"],
        "SITE_NO": [123, 123, 456],
        "EASTING_GDA2020": [514129.3, 514129.3, 600000.0],
        "NORTHING_GDA2020": [6681073.2, 6681073.2, 6700000.0],
        "chem_code": ["Cu", "Au", "Zn"],
        "value": [85.0, 0.02, 120.0],
        "value_text": ["85", "0.02", "120"],
        "unit": ["ppm", "ppm", "ppm"],
    })


def test_identifier_file_date_coordinate_and_numeric_columns_are_skipped():
    candidates = _candidate_columns(_resolvesa_like())
    skipped = [
        "filename", "drillhole_number", "core", "LOG_NUMBER", "LOGGING_DATE", "depth_from_m", "depth_to_m",
        "SITE_NO", "EASTING_GDA2020", "NORTHING_GDA2020", "value", "value_text",
    ]
    for col in skipped:
        assert col not in candidates, col
    for col in ["LOGGING_ORGANISATION", "major_lithology", "DESCRIPTION", "chem_code", "unit"]:
        assert col in candidates, col


def test_name_rules_do_not_catch_ordinary_words():
    df = pd.DataFrame({
        "weathering_profile": ["fresh", "oxidised"],
        "pathfinder_notes": ["As anomaly", "none"],
        "validated_by": ["JS", "AB"],
    })
    assert _candidate_columns(df) == ["weathering_profile", "pathfinder_notes", "validated_by"]


def test_pick_refuses_when_only_ids_and_numbers(workspace):
    path = workspace / "ids_only.csv"
    pd.DataFrame({
        "filename": ["a_1.jpg", "b_2.jpg"],
        "drillhole_number": [1, 2],
        "EASTING": [500000.0, 500010.0],
        "NORTHING": [6600000.0, 6600010.0],
        "value": [1.5, 2.5],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Convert to Voxel"):
        pick_text_column(str(path), "Cu in the file")


@pytest.mark.parametrize("field", ["filename", "drillhole_number", "value"])
def test_chat_tools_refuse_non_text_fields(workspace, field):
    import tools.text_classify as text_classify

    path = workspace / "segments.csv"
    _resolvesa_like().to_csv(path, index=False)
    out = asyncio.run(text_classify.classify_text_discover(str(path), field, "minerals mentioned"))
    assert out["success"] is False
    assert "not a free-text column" in out["error"]
    assert "DESCRIPTION" in out["error"]
