"""Text row -> category value via LLM, then stamped into a voxel layer.

Point 3 of the original ask: numeric+coordinate data can be coloured
without AI (see csv_to_voxel.py); free-text data needs an LLM to first turn
it into a category per row before it can be visualised the same way.
Categories aren't limited to yes/no -- pass any list of 2+ labels and the
model picks one per row (voxel_upsert_geometry's categorical dtype already
supports arbitrary label counts, this was just never wired up to more than
two before).

The LLM call goes straight to this IDE's LLM proxy (LLM_PROXY_URL +
IDE_LLM_TOKEN -- a short-lived, project-scoped JWT, not the real API key;
entrypoint.sh deliberately leaves these two unscrubbed) via a plain HTTP
POST. No MCP protocol, no chat agent deciding what happens next -- the only
thing the model does here is pick one label per row. Model is caller-chosen
(any OpenRouter model id); defaults to DEFAULT_MODEL if not given.

Usage (binary):
    classify_text_to_voxel("ReSolveSA/segments_with_geochem.csv",
                            text_col="DESCRIPTION",
                            question="does this mention visible sulphide mineralisation?",
                            categories=["no", "yes"], layer="sulphide_mentioned")

Usage (multi-category):
    classify_text_to_voxel(..., question="what is the dominant lithology described?",
                            categories=["sandstone", "shale", "limestone", "other"],
                            layer="lithology_from_text")
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from direct._log import get_logger
from direct.csv_to_voxel import _current_layer_names, _project_lock, geometry_records
from tools.config import read_tabular, resolve_path, save_csv

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BATCH_SIZE = 25
MAX_CATEGORIES = 20
_log = get_logger("classify_text_to_voxel")


def _proxy_chat(messages: list[dict], model: str | None) -> str:
    import httpx

    proxy_url = os.environ.get("LLM_PROXY_URL")
    token = os.environ.get("IDE_LLM_TOKEN")
    if not proxy_url or not token:
        raise RuntimeError("LLM_PROXY_URL / IDE_LLM_TOKEN are not set in this environment")

    resp = httpx.post(
        f"{proxy_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "model": model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _slug(text: str, max_len: int = 30) -> str:
    """Turn free text into something validate_layer_name() accepts:
    ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$ -- so spaces and punctuation become
    underscores. Used for both user-typed prefixes and AI-returned labels.
    """
    s = "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in text.lower().strip())
    return s.strip("_")[:max_len]


def _candidate_columns(df) -> list[str]:
    """Every column that isn't a coordinate/depth column -- same exclusion
    csv_to_voxel's inspect uses, so "pick a column" and "convert to voxel"
    agree on what counts as a real data column.
    """
    import re

    from direct.csv_to_voxel import EASTING_PATTERNS, LAT_PATTERNS, LON_PATTERNS, NORTHING_PATTERNS, depth_columns

    dc = depth_columns(list(df.columns))
    geo_patterns = EASTING_PATTERNS + NORTHING_PATTERNS + LON_PATTERNS + LAT_PATTERNS
    structural = {c for c in df.columns if any(re.search(p, str(c).strip().lower()) for p in geo_patterns)}
    structural |= {dc.get("depth"), dc.get("from"), dc.get("to")}
    structural -= {None}
    return [c for c in df.columns if c not in structural]


def pick_text_column(dataset_path: str, goal: str, model: str | None = None) -> dict[str, Any]:
    """Ask the LLM which column to read to accomplish ``goal`` -- the user
    doesn't pick the text column by hand. Each candidate column gets a few
    real example values (not just its name) so the model can see what's
    actually in it.
    """
    resolved = resolve_path(dataset_path)
    df = read_tabular(resolved, nrows=200)
    candidates = _candidate_columns(df)
    if not candidates:
        raise ValueError(f"no candidate columns in {dataset_path} (everything looks like a coordinate/depth column)")

    lines = []
    for c in candidates:
        samples = [str(v) for v in df[c].dropna().astype(str).unique()[:3]]
        lines.append(f"- {c}: e.g. {', '.join(repr(s)[:80] for s in samples) if samples else '(no sample values)'}")
    cols_desc = "\n".join(lines)

    prompt = (
        f"A spreadsheet has these columns:\n{cols_desc}\n\n"
        f"Which ONE column should be read, row by row, to do this: {goal}\n\n"
        'Reply with ONLY a JSON object {"column": "<exact column name from the list above>"}.'
    )
    content = _proxy_chat([{"role": "user", "content": prompt}], model)
    parsed = json.loads(content)
    picked = parsed["column"]
    if picked not in candidates:
        raise ValueError(f"model picked {picked!r}, which isn't one of the candidate columns: {candidates}")
    return {"success": True, "column": picked, "candidates": candidates}


DEFAULT_DISCOVERY_SAMPLE_SIZE = 200


def discover_categories(
    dataset_path: str,
    text_col: str,
    goal: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    sample_size: int = DEFAULT_DISCOVERY_SAMPLE_SIZE,
    model: str | None = None,
    max_categories: int | None = None,
) -> dict[str, Any]:
    """For an "incomplete" file with no reliable category column: sample
    (deduplicated) values from ``text_col`` and ask the LLM what distinct
    things are mentioned that satisfy ``goal`` -- e.g. goal="minerals
    mentioned" over a free-text DESCRIPTION column with no chem_code. This
    does NOT read every row (that would need ~rows/batch_size calls just to
    find out what to look for); it is a cheap first pass that returns
    *candidate* labels for the caller to run real per-row classification
    against afterwards.

    ``max_categories`` is unlimited by default: each label costs a full
    classification pass later, but the caller (the IDE shows a checkbox list
    and a time estimate) is better placed to decide what to keep than an
    arbitrary cap here.
    """
    resolved = resolve_path(dataset_path)
    df = read_tabular(resolved)

    if filter_col:
        if filter_col not in df.columns:
            raise ValueError(f"filter_col {filter_col!r} not found in {dataset_path}")
        df = df[df[filter_col].astype(str) == str(filter_value)]

    if text_col not in df.columns:
        raise ValueError(f"text_col {text_col!r} not found in {dataset_path}")

    texts = df[text_col].dropna().astype(str)
    texts = texts[texts.str.strip() != ""].drop_duplicates()
    if texts.empty:
        raise ValueError(f"no non-empty text in {text_col!r} to sample from")

    sample = texts.sample(n=min(sample_size, len(texts)), random_state=0).tolist()
    numbered = "\n".join(f"- {t!r}" for t in sample)

    cap_line = (
        f"Return at most {max_categories} entries, most common first."
        if max_categories
        else "List every distinct one you find, most common first."
    )
    prompt = (
        f"Here are {len(sample)} distinct example values from a \"{text_col}\" column "
        f"(out of {len(texts)} distinct values total, {len(df)} rows before dedup):\n{numbered}\n\n"
        f"Goal: {goal}\n\n"
        f"List the distinct things mentioned across these examples that are relevant to the goal "
        f"(e.g. distinct mineral names). Merge obvious synonyms/variants into one entry. "
        f"{cap_line}\n\n"
        'Reply with ONLY a JSON object {"categories": ["...", "..."]}.'
    )
    content = _proxy_chat([{"role": "user", "content": prompt}], model)
    parsed = json.loads(content)
    categories = [str(c).strip() for c in parsed["categories"] if str(c).strip()]
    categories = list(dict.fromkeys(categories))  # dedupe, keep order
    if max_categories:
        categories = categories[:max_categories]
    if not categories:
        raise ValueError("model didn't find anything relevant in the sample")

    return {
        "success": True,
        "categories": categories,
        "sample_size": len(sample),
        "distinct_text_values": len(texts),
        "rows_before_dedup": int(len(df)),
    }


def classify_multi_label_to_voxel(
    dataset_path: str,
    text_col: str,
    goal: str,
    categories: list[str],
    layer_prefix: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    combination_rule: str = "max",
) -> dict[str, Any]:
    """Given an already-decided ``categories`` list (from discover_categories,
    possibly trimmed by the caller), run one *separate binary*
    classify_text_to_voxel() per label -- "does this row mention <label>?" --
    instead of one multi-choice pass. That's what makes it multi-label: a
    row can end up "yes" in several of the resulting layers at once (e.g. a
    description mentioning both pyrite and chalcopyrite), which a single
    categorical layer (one value per voxel) can't represent, but N binary
    layers can.
    """
    _log.info("classify_multi_label_to_voxel start: dataset=%s text_col=%s layer_prefix=%s categories=%s layers_before=%s",
              dataset_path, text_col, layer_prefix, categories, _current_layer_names())
    # The prefix comes straight from a user input box, so sanitise it the same
    # way as the category slug -- a prefix like "all mineral" would otherwise
    # produce "all mineral_copper", which validate_layer_name() rejects only
    # after every AI call for that category has already been paid for.
    prefix = _slug(layer_prefix, 30)
    layers: dict[str, Any] = {}
    for cat in categories:
        slug = _slug(cat, 30)
        layer = f"{prefix}_{slug}" if prefix else slug
        try:
            layers[cat] = classify_text_to_voxel(
                dataset_path, text_col=text_col, question=f"does this mention \"{cat}\"? (context: {goal})",
                categories=["no", "yes"], layer=layer, filter_col=filter_col, filter_value=filter_value,
                model=model, batch_size=batch_size, combination_rule=combination_rule,
            )
        except Exception as exc:  # noqa: BLE001
            _log.exception("classify_text_to_voxel(%s) raised", layer)
            layers[cat] = {"success": False, "error": str(exc)}

    ok = sum(1 for v in layers.values() if v.get("success"))
    _log.info("classify_multi_label_to_voxel done: %d/%d succeeded, layers_after=%s",
              ok, len(categories), _current_layer_names())

    return {
        "success": all(v.get("success") for v in layers.values()),
        "categories": categories,
        "layers": layers,
    }


def discover_and_classify_text_to_voxel(
    dataset_path: str,
    text_col: str,
    goal: str,
    layer_prefix: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    combination_rule: str = "max",
) -> dict[str, Any]:
    """discover_categories() then classify_multi_label_to_voxel() on
    everything it found, in one call -- convenience for CLI/script use.
    The IDE button calls the two steps separately instead, so it can show
    the discovered list and let the user drop ones they don't want before
    the (more expensive) classification passes run.
    """
    discovery = discover_categories(
        dataset_path, text_col, goal, filter_col=filter_col, filter_value=filter_value, model=model
    )
    result = classify_multi_label_to_voxel(
        dataset_path, text_col, goal, discovery["categories"], layer_prefix,
        filter_col=filter_col, filter_value=filter_value, model=model,
        batch_size=batch_size, combination_rule=combination_rule,
    )
    result["discovery"] = {k: v for k, v in discovery.items() if k != "categories"}
    return result


def _classify_batch(texts: list[str], question: str, categories: list[str], model: str | None) -> list[int]:
    """One LLM call for up to a few dozen rows at once (cheaper and faster
    than one call per row). Returns a category INDEX per row. Raises on any
    failure to parse or an out-of-range index -- the caller treats the whole
    batch as unclassified rather than guess.
    """
    cat_list = "\n".join(f"{i}: {c}" for i, c in enumerate(categories))
    numbered = "\n".join(f"{i}: {t!r}" for i, t in enumerate(texts))
    prompt = (
        f"{question}\n\n"
        f"Pick exactly one category per text, by number:\n{cat_list}\n\n"
        f"Texts:\n{numbered}\n\n"
        'Reply with ONLY a JSON object of the form {"answers": [...]}, one category number per '
        f"line in order. The \"answers\" array must have exactly {len(texts)} entries, each an "
        f"integer from 0 to {len(categories) - 1}."
    )
    content = _proxy_chat([{"role": "user", "content": prompt}], model)
    parsed = json.loads(content)
    answers = parsed["answers"]
    if len(answers) != len(texts):
        raise ValueError(f"model returned {len(answers)} answers for {len(texts)} rows")
    indices = [int(a) for a in answers]
    if any(not (0 <= i < len(categories)) for i in indices):
        raise ValueError(f"model returned an out-of-range category index (expected 0-{len(categories) - 1})")
    return indices


def _classify_batch_with_retry(
    texts: list[str], question: str, categories: list[str], model: str | None, retries: int = 1
) -> list[int]:
    """_classify_batch(), retrying transient failures (a proxy 502/timeout,
    or a model returning unparseable JSON) a couple of times before giving
    up -- free/shared-capacity models in particular are prone to occasional
    gateway errors that succeed on a plain retry.
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _classify_batch(texts, question, categories, model)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def classify_column(
    dataset_path: str,
    text_col: str,
    question: str,
    categories: list[str],
    filter_col: str | None = None,
    filter_value: str | None = None,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Ask the LLM proxy ``question`` about each row's ``text_col``, in
    batches, picking one of ``categories`` per row. Returns the (possibly
    filtered) dataframe, a category-label value series aligned to it (NaN =
    empty text or a batch that failed to parse), and stats.
    """
    import pandas as pd

    if len(categories) < 2:
        raise ValueError("need at least 2 categories")
    if len(categories) > MAX_CATEGORIES:
        raise ValueError(f"too many categories ({len(categories)}); max {MAX_CATEGORIES}")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique")

    resolved = resolve_path(dataset_path)
    df = read_tabular(resolved)

    if filter_col:
        if filter_col not in df.columns:
            raise ValueError(f"filter_col {filter_col!r} not found in {dataset_path}")
        df = df[df[filter_col].astype(str) == str(filter_value)].copy()
        if df.empty:
            raise ValueError(f"no rows where {filter_col} == {filter_value!r}")

    if text_col not in df.columns:
        raise ValueError(f"text_col {text_col!r} not found in {dataset_path}")

    texts_series = df[text_col].fillna("").astype(str)
    # Classify each DISTINCT text value once, not once per row -- geoscience
    # tables like this commonly repeat the same free-text value across many
    # rows (e.g. one drillhole interval logged once but joined to several
    # chem_code rows), so this can turn thousands of calls into a handful.
    unique_texts = [t for t in texts_series.unique().tolist() if t.strip()]
    unique_labels: dict[str, str | None] = {t: None for t in unique_texts}
    n_batch_failures = 0
    n_batches = math.ceil(len(unique_texts) / batch_size) if unique_texts else 0

    for start in range(0, len(unique_texts), batch_size):
        chunk = unique_texts[start : start + batch_size]
        try:
            result = _classify_batch_with_retry(chunk, question, categories, model)
        except Exception:  # noqa: BLE001
            n_batch_failures += 1
            continue
        for t, idx in zip(chunk, result):
            unique_labels[t] = categories[idx]

    value = texts_series.map(lambda t: unique_labels.get(t) if t.strip() else None)
    value = pd.Series(value, index=df.index)

    return {
        "df": df,
        "value": value,
        "stats": {
            "question": question,
            "categories": categories,
            "rows_in": int(len(texts_series)),
            "distinct_text_values": len(unique_texts),
            "rows_classified": int(value.notna().sum()),
            "rows_empty_text": int((texts_series.str.strip() == "").sum()),
            "n_batches": n_batches,
            "n_batch_failures": n_batch_failures,
        },
    }


def classify_text_to_voxel(
    dataset_path: str,
    text_col: str,
    question: str,
    categories: list[str],
    layer: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    combination_rule: str = "max",
) -> dict[str, Any]:
    """End to end: ask the LLM about each row -> stamp the chosen category
    into the project's voxel store as a categorical layer -> export viz/.
    Shares its grid and export logic with csv_to_voxel() so an LLM-labelled
    layer and a plain numeric one from the same project can sit in the same
    view. Category codes are fixed by ``categories`` order (0, 1, 2, ...),
    passed as label_map so the manifest carries real names, not just numbers.
    """
    from tools.voxel import voxel_export_bundle, voxel_get_grid, voxel_init_grid, voxel_upsert_geometry

    _log.info("classify_text_to_voxel start: dataset=%s text_col=%s layer=%s categories=%s",
              dataset_path, text_col, layer, categories)

    # Check the name BEFORE any AI call -- voxel_upsert_geometry validates it
    # too, but that happens after the whole column has been classified, so a
    # bad name would otherwise cost a full (paid) classification pass first.
    from voxel.store import validate_layer_name

    validate_layer_name(layer)

    with _project_lock():
        classified = classify_column(
            dataset_path, text_col, question, categories, filter_col=filter_col, filter_value=filter_value,
            model=model, batch_size=batch_size,
        )
        geo = geometry_records(classified["df"], dataset_path, classified["value"])
        records_path = save_csv(geo["records"], dataset_path, f"voxel_records_{layer}")

        grid_result = voxel_get_grid()
        if not grid_result.get("success"):
            _log.info("no existing grid (%s) -- calling voxel_init_grid", grid_result.get("error"))
            grid_result = voxel_init_grid(dataset_path=dataset_path)
            _log.info("voxel_init_grid -> success=%s layers_now=%s", grid_result.get("success"), _current_layer_names())
            if not grid_result.get("success"):
                return {"success": False, "step": "voxel_init_grid", "classification": classified["stats"], **grid_result}

        label_map = {cat: i for i, cat in enumerate(categories)}
        upsert_result = voxel_upsert_geometry(
            layer=layer, records_path=records_path, value_col="value", dtype="categorical",
            label_map=label_map, combination_rule=combination_rule, source_file=dataset_path, source_check="reject",
            max_records=len(geo["records"]),  # default (5000) silently drops rows past it on bigger datasets
        )
        _log.info("voxel_upsert_geometry(%s) -> success=%s applied=%s layers_now=%s",
                  layer, upsert_result.get("success"), upsert_result.get("records_applied"), _current_layer_names())
        if not upsert_result.get("success"):
            return {"success": False, "step": "voxel_upsert_geometry", "classification": classified["stats"], **upsert_result}

        export_result = voxel_export_bundle()
        _log.info("voxel_export_bundle -> success=%s layer_ids=%s",
                  export_result.get("success"), export_result.get("layer_ids"))

        return {
            "success": bool(export_result.get("success")),
            "classification": {**classified["stats"], "geometry_kind": geo["geometry_kind"]},
            "grid": grid_result.get("grid"),
            "upsert": {k: v for k, v in upsert_result.items() if k != "warnings"},
            "export": export_result,
        }


def _cli() -> None:
    """Subprocess entrypoint (mirrors direct/csv_to_voxel.py's CLI):

      python3 -m direct.classify_text_to_voxel classify --dataset X --text-col C
                                                --question Q --category A --category B [--category C ...]
                                                --layer L [--model M]
                                                [--filter-col FC --filter-value FV]
    """
    import argparse
    import sys

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pick = sub.add_parser("pick-column")
    pick.add_argument("--dataset", required=True)
    pick.add_argument("--goal", required=True)
    pick.add_argument("--model", default=None)

    disc = sub.add_parser("discover")
    disc.add_argument("--dataset", required=True)
    disc.add_argument("--text-col", required=True)
    disc.add_argument("--goal", required=True)
    disc.add_argument("--filter-col", default=None)
    disc.add_argument("--filter-value", default=None)
    disc.add_argument("--model", default=None)
    disc.add_argument("--max-categories", type=int, default=None,
                       help="cap the list (default: no cap -- the caller decides what to keep)")

    cls = sub.add_parser("classify")
    cls.add_argument("--dataset", required=True)
    cls.add_argument("--text-col", required=True)
    cls.add_argument("--question", required=True)
    cls.add_argument("--category", action="append", dest="categories", required=True,
                      help="repeatable; pass at least twice, e.g. --category no --category yes")
    cls.add_argument("--layer", required=True)
    cls.add_argument("--filter-col", default=None)
    cls.add_argument("--filter-value", default=None)
    cls.add_argument("--model", default=None)

    disc_cls = sub.add_parser("discover-classify")
    disc_cls.add_argument("--dataset", required=True)
    disc_cls.add_argument("--text-col", required=True)
    disc_cls.add_argument("--goal", required=True)
    disc_cls.add_argument("--layer-prefix", required=True)
    disc_cls.add_argument("--filter-col", default=None)
    disc_cls.add_argument("--filter-value", default=None)
    disc_cls.add_argument("--model", default=None)

    multi = sub.add_parser("classify-multi")
    multi.add_argument("--dataset", required=True)
    multi.add_argument("--text-col", required=True)
    multi.add_argument("--goal", required=True)
    multi.add_argument("--category", action="append", dest="categories", required=True,
                        help="repeatable; the (possibly user-trimmed) list from `discover`")
    multi.add_argument("--layer-prefix", required=True)
    multi.add_argument("--filter-col", default=None)
    multi.add_argument("--filter-value", default=None)
    multi.add_argument("--model", default=None)

    args = p.parse_args()

    try:
        if args.mode == "pick-column":
            result = pick_text_column(args.dataset, args.goal, model=args.model)
        elif args.mode == "discover":
            result = discover_categories(
                args.dataset, args.text_col, args.goal,
                filter_col=args.filter_col, filter_value=args.filter_value, model=args.model,
                max_categories=args.max_categories,
            )
        elif args.mode == "discover-classify":
            result = discover_and_classify_text_to_voxel(
                args.dataset, text_col=args.text_col, goal=args.goal, layer_prefix=args.layer_prefix,
                filter_col=args.filter_col, filter_value=args.filter_value, model=args.model,
            )
        elif args.mode == "classify-multi":
            result = classify_multi_label_to_voxel(
                args.dataset, text_col=args.text_col, goal=args.goal, categories=args.categories,
                layer_prefix=args.layer_prefix, filter_col=args.filter_col, filter_value=args.filter_value,
                model=args.model,
            )
        else:
            result = classify_text_to_voxel(
                args.dataset, text_col=args.text_col, question=args.question, categories=args.categories,
                layer=args.layer, filter_col=args.filter_col, filter_value=args.filter_value, model=args.model,
            )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)

    print(json.dumps(result))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    _cli()
