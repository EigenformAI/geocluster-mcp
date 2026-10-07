"""Package exact engine registry entries; run with the owner's generated registry path.

No engine schema is authored here. The source digest identifies the complete
registry; entries remain byte-semantically identical after JSON serialization.
"""
import hashlib
import json
import sys
from pathlib import Path

NAMES = {"session", "get", "explain", "query", "compare_commits", "list_operations", "plan_recipe",
         "preview_operations", "apply", "ingest_sources", "submit_job", "poll_job", "job_result",
         "cancel_job", "fetch_artifact", "rank_hypothesis_set"}


def main():
    raw = Path(sys.argv[1]).read_bytes()
    registry = json.loads(raw)
    operations = [op for op in registry["operations"] if op["name"] in NAMES]
    assert {op["name"] for op in operations} == NAMES
    output = {"schema": registry["schema"], "source_sha256": hashlib.sha256(raw).hexdigest(), "operations": operations}
    target = Path(__file__).with_name("geomodel-registry.json")
    text = json.dumps(output, separators=(",", ":"), ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        assert target.read_text() == text, "packaged engine registry is stale"
    else:
        target.write_text(text)
    print(f"geomodel registry: {len(operations)} exact engine entries, sha256={output['source_sha256']}")


if __name__ == "__main__":
    main()
