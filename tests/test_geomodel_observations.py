import asyncio
import copy
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP
from jsonschema import Draft7Validator

from tools import config, geomodel, runctx

BINDING = {"projectId": "p", "scientificProjectId": "s", "bindingEpoch": "e"}
JOB = {"job_id": "job-original", "principal": "owner", "session_id": "owner-session", "request_digest": "digest", "status": "queued", "submitted_at": "now"}


@pytest.fixture
def science(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(runctx, "SHARED", True)
    token = tmp_path / "control"
    token.write_text("private-control")
    monkeypatch.setenv("SCIENTIFIC_RUNTIME_URL", "http://127.0.0.1:3001/internal/science")
    monkeypatch.setenv("SCIENTIFIC_RUNTIME_TOKEN_FILE", str(token))
    server = FastMCP("science")
    server.add_middleware(runctx.RunDirMiddleware())
    geomodel.register_tools(server)
    return server


def scope(args):
    return {**args, "run_dir": "runs/ses_root", "actingSessionId": "ses_child", "rootConversationId": "ses_root", "invocationId": "call-1"}


def envelope(value, intent="original-intent"):
    return {"schema": "geocluster.science/1", "binding": BINDING, "intentId": intent, "result": value}


def observe(operation="apply", state="unknown", result=None, intent="original-intent"):
    return {"intentId": intent, "operation": operation, "state": state, "binding": BINDING, "result": result,
            "control": "available", "observedAt": 1, "freshness": "current", "payloadDigest": "digest"}


def call(server, operation, args):
    async def journey():
        async with Client(server) as client:
            result = await client.call_tool(f"geomodel_{operation}", scope(args), raise_on_error=False)
            if result.structured_content:
                listed = {t.name: t for t in await client.list_tools()}
                Draft7Validator(listed[f"geomodel_{operation}"].outputSchema).validate(result.structured_content)
            return result
    return asyncio.run(journey())


@pytest.mark.parametrize("state", ["recorded", "unknown", "forwarding"])
def test_pending_mutation_is_valid_progress_not_engine_success(science, monkeypatch, state):
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=envelope(observe(state=state))))))
    result = call(science, "apply", {"batch": {"operations": []}, "command_id": "c1", "idempotency_key": "stable"})
    assert not result.is_error
    assert result.structured_content["result"]["kind"] == "intent_observation"
    assert result.structured_content["result"]["state"] == state
    assert result.structured_content["result"]["result"] is None
    assert result.structured_content["metadata"]["intentId"] == "original-intent"


@pytest.mark.parametrize("change", [{"intentId": "forged"}, {"binding": {**BINDING, "bindingEpoch": "old"}}, {"operation": "submit_job"}, {"state": "succeeded"}, {"state": "ready"}, {"result": {"job_id": "fake"}}, {"state": "accepted", "result": {"job_id": "fake"}}])
def test_pending_does_not_bypass_engine_validation(science, monkeypatch, change):
    value = {**observe(), **change}
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=envelope(value)))))
    result = call(science, "apply", {"batch": {"operations": []}, "command_id": "c1", "idempotency_key": "stable"})
    assert result.is_error


def test_read_intent_normalizes_verified_original_job(science, monkeypatch):
    value = {**observe("submit_job", "accepted", JOB), "execution": {"job": JOB}}
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=envelope(value)))))
    result = call(science, "read_intent", {"intent_id": "original-intent"})
    assert not result.is_error
    assert result.structured_content["result"] == JOB
    assert result.structured_content["metadata"]["originalOperation"] == "submit_job"
    assert result.structured_content["metadata"]["observation"]["progress"]["execution"]["job"]["status"] == "queued"


@pytest.mark.parametrize("operation,state", [("ingest_sources", "uploading"), ("prepare_review", "preparing")])
def test_transfer_and_review_progress_keep_original_intent(science, monkeypatch, operation, state):
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=envelope(observe(operation, state))))))
    result = call(science, "read_intent", {"intent_id": "original-intent"})
    assert not result.is_error
    assert result.structured_content["result"]["operation"] == operation
    assert result.structured_content["result"]["state"] == state
    assert result.structured_content["result"]["result"] is None


def test_nested_scientific_evidence_is_not_request_authority(science, monkeypatch):
    received = []
    def handler(request):
        received.append(json.loads(request.content)["request"])
        return httpx.Response(400, json={**envelope(None), "error": {"code": "recipe.unavailable", "message": "No such recipe."}})
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    evidence = {"source": {"session_id": "historical-capture", "principal": "source-author"}}
    result = call(science, "plan_recipe", {"args": evidence, "base_commit": "commit", "recipe": "unsupported"})
    assert result.is_error
    assert result.structured_content["error"]["code"] == "recipe.unavailable"
    assert received[0]["args"] == evidence


def test_session_returns_actual_owner_budget(science, monkeypatch):
    schema = geomodel.engine_operations()["session"]["response_schema"]
    amounts = dict.fromkeys(schema["definitions"]["SessionResourceAmountsV0"]["required"], 0)
    session = {"budget": {"job_count": {"hard": True, "limit": 7}}, "budget_usage": {"consumed": amounts, "outstanding": amounts}, "budgets_remaining": {}, "capabilities": ["job:submit"], "client_kind": "mcp", "created_at": 1, "expires_at": 100, "principal": "owner", "principal_kind": "service", "read_scope": {"scope": "project"}, "rejected_submission_count": 0, "session_id": "owner-session"}
    seen = []
    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json=envelope({"available": True, "operations": ["session"]} if request.method == "GET" else session))
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = call(science, "session", {})
    assert not result.is_error
    assert result.structured_content["result"]["session"]["budget"]["job_count"]["limit"] == 7
    assert seen == ["GET", "POST"]


def test_unavailable_capabilities_report_unknown_session_facts(science, monkeypatch):
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(503, json={**envelope({"available": False}), "error": {"code": "science_unavailable", "message": "Owner unavailable."}})
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = call(science, "session", {})
    assert not result.is_error
    assert result.structured_content["result"]["sessionAvailability"] == "unavailable"
    assert result.structured_content["result"]["session"] is None
    assert result.structured_content["result"]["sessionError"]["code"] == "science_unavailable"


def test_explicit_artifact_page_contains_actual_bytes_no_product_locator(science, monkeypatch):
    artifact = {"byte_count": 3, "data_hex": "010203", "descriptor": {"digest": "digest", "media_type": "application/octet-stream", "role": "prediction", "schema_profile": "array/0", "size_bytes": 3}, "exhaustive": True, "offset": 0, "producers": [], "producers_exhaustive": True, "producing_job_id": "job-original", "read_receipt_id": "read-1", "representation_revisions": []}
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=envelope(artifact)))))
    result = call(science, "fetch_artifact", {"digest": "digest", "offset": 0, "max_bytes": 3})
    assert not result.is_error
    assert result.structured_content["result"] == artifact
    assert "010203" not in result.content[0].text
    assert "/science/products/" not in json.dumps(result.structured_content)


def test_actual_runtime_science_http_pending_resume_and_expired_poll(science, tmp_path, monkeypatch):
    env = dict(os.environ)
    fixture = Path(__file__).with_name("science_http_fixture.mjs")
    process = subprocess.Popen(["node", str(fixture), str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        ready = process.stdout.readline()
        assert ready, process.stderr.read()
        monkeypatch.setenv("SCIENTIFIC_RUNTIME_URL", json.loads(ready)["url"])
        args = {"batch": {"operations": []}, "command_id": "c1", "idempotency_key": "stable"}
        first = call(science, "apply", copy.deepcopy(args))
        assert not first.is_error, first.content
        intent = first.structured_content["metadata"]["intentId"]
        assert first.structured_content["result"]["state"] == "unknown"
        resumed = call(science, "apply", copy.deepcopy(args))
        assert not resumed.is_error
        assert resumed.structured_content["metadata"]["intentId"] == intent
        assert resumed.structured_content["result"]["receipt_id"] == "r1"
        job = call(science, "read_intent", {"intent_id": "submission-intent"})
        assert not job.is_error and job.structured_content["result"] == JOB
        wrong = call(science, "read_intent", {"intent_id": "wrong-intent"})
        assert wrong.is_error
        expired = call(science, "poll_job", {"job_id": "job-original", "wait_ms": 0})
        assert expired.is_error
        assert expired.structured_content["error"]["code"] == "session.original-unavailable"
        assert expired.structured_content["metadata"]["observation"]["freshness"] == "unavailable"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def supervisory(operation, terminal=False):
    job = {**JOB, "status": "failed" if terminal else "queued"}
    receipt = None
    if terminal:
        job["receipt_id"] = "failed-not-started"
        charge = {"basis": "not-started", "seconds": 0}
        receipt = {"job_id": JOB["job_id"], "receipt_id": "failed-not-started", "status": "failed", "recorded_at": "now",
            "artifacts": [], "diagnostics": [{"validator_id": "session.original-unavailable", "message": "Failed — execution session unavailable"}],
            "ref_update": {"outcome": "not_requested"}, "settlement": {"cpu": charge, "wall_time": charge, "usage_evidence": [], "usage_evidence_digests": []}}
    return {"schema": "geocluster.science.job-observation/1", "operation": operation, "job": job, "receipt": receipt,
        "events": [{"cursor": 1, "message": "Original job observed", "recorded_at": "now", "status": job["status"]}],
        "eventsExhaustive": True, "nextEventCursor": None, "control": "session-unavailable", "readReceiptId": None}


@pytest.mark.parametrize("operation,terminal", [("poll_job", False), ("cancel_job", True)])
def test_supervisory_original_job_is_valid_without_fabricated_engine_read(science, monkeypatch, operation, terminal):
    value = supervisory(operation, terminal)
    # Supervisory observations have no integration intent identity.
    response = {"schema": "geocluster.science/1", "binding": BINDING, "result": value}
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))))
    result = call(science, operation, {"job_id": JOB["job_id"], **({"wait_ms": 0} if operation == "poll_job" else {"reason": "Stop the original job"})})
    assert not result.is_error
    assert result.structured_content["result"] == value
    assert result.structured_content["result"]["job"]["status"] == ("failed" if terminal else "queued")
    assert result.structured_content["result"]["readReceiptId"] is None
    assert "outcome" not in result.structured_content["result"]


@pytest.mark.parametrize("defect", ["job", "operation", "receipt-job", "receipt-status", "read-receipt", "event"])
def test_supervisory_rejects_foreign_job_operation_receipt_and_forged_reads(science, monkeypatch, defect):
    value = supervisory("cancel_job", True)
    if defect == "job": value["job"]["job_id"] = "different-job"
    elif defect == "operation": value["operation"] = "poll_job"
    elif defect == "receipt-job": value["receipt"]["job_id"] = "different-job"
    elif defect == "receipt-status": value["receipt"]["status"] = "cancelled"
    elif defect == "read-receipt": value["readReceiptId"] = "fabricated-read"
    else: value["events"][0]["cursor"] = -1
    response = {"schema": "geocluster.science/1", "binding": BINDING, "result": value}
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))))
    assert call(science, "cancel_job", {"job_id": JOB["job_id"], "reason": "Stop the original job"}).is_error


def test_supervisory_optional_events_are_not_synthesized(science, monkeypatch):
    value = supervisory("poll_job")
    for key in ("events", "eventsExhaustive", "nextEventCursor"):
        del value[key]
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"schema": "geocluster.science/1", "binding": BINDING, "result": value}))))
    result = call(science, "poll_job", {"job_id": JOB["job_id"], "wait_ms": 0})
    assert not result.is_error
    assert result.structured_content["result"] == value


def test_all_engine_request_authority_nodes_are_covered_without_semantic_stripping():
    operations = geomodel.engine_operations()
    assert len(operations) == 16
    definitions_with_authority = set()
    def dereference(schema, node):
        while "$ref" in node:
            assert node["$ref"].startswith("#/definitions/")
            node = schema["definitions"][node["$ref"].split("/")[-1]]
        return node
    for name, operation in operations.items():
        original = operation["request_schema"]
        adapted = geomodel.input_schema("compare" if name == "compare_commits" else name)
        root = dereference(original, original)
        public_root = dereference(adapted, adapted)
        assert not geomodel.HOST_FIELDS & public_root["properties"].keys(), name
        assert not geomodel.HOST_FIELDS & set(public_root.get("required", [])), name
        for key, definition in original.get("definitions", {}).items():
            if geomodel.HOST_FIELDS & definition.get("properties", {}).keys():
                definitions_with_authority.add(key)
                assert key == "Command", (name, key)
                public = dereference(adapted, root["properties"]["command"])
                assert not geomodel.HOST_FIELDS & public["properties"].keys()
                assert not geomodel.HOST_FIELDS & set(public.get("required", []))
            else:
                assert adapted["definitions"][key] == definition, (name, key)
    assert definitions_with_authority == {"Command"}
