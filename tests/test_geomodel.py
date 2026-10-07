"""Scientific facade: engine schemas, private delegation, stable action identity."""
import asyncio
import copy
import json

import httpx
import pytest
from fastmcp import Client, FastMCP
from jsonschema import Draft7Validator, ValidationError

from tools import config, runctx


def test_inventory_and_engine_schema():
    from tools.geomodel import OPERATIONS, engine_operations, input_schema, output_schema

    assert len(OPERATIONS) == 19
    original = engine_operations()["submit_job"]["request_schema"]
    schema = input_schema("submit_job")
    assert schema["properties"]["specification"] == original["properties"]["specification"]
    assert schema["definitions"] == original["definitions"]
    assert not {"principal", "session_id", "token"} & schema["properties"].keys()
    assert "stable" in schema["properties"]["idempotency_key"]["description"]
    for operation in OPERATIONS:
        Draft7Validator.check_schema(input_schema(operation))
        Draft7Validator.check_schema(output_schema(operation))


def test_response_loss_reuses_original_action_and_structured_result(tmp_path, monkeypatch):
    from tools import geomodel

    monkeypatch.setattr(config, "WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(runctx, "SHARED", True)
    token = tmp_path / "control"
    token.write_text("private-control")
    monkeypatch.setenv("SCIENTIFIC_RUNTIME_URL", "http://127.0.0.1:3001/internal/science")
    monkeypatch.setenv("SCIENTIFIC_RUNTIME_TOKEN_FILE", str(token))
    calls = []
    receipt = {"command_id": "c1", "diagnostics": [], "payload_digest": "digest", "receipt_id": "r1",
               "recorded_at": "now", "status": "applied", "validations_passed": [], "written_revisions": []}

    def handler(request):
        assert request.headers["x-gc-runtime-control"] == "private-control"
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            raise httpx.ReadError("response lost")
        if len(calls) == 3:
            return httpx.Response(409, json={"schema": "geocluster.science/1", "binding": {
                "projectId": "p", "scientificProjectId": "s", "bindingEpoch": "e"}, "intentId": "original-intent",
                "error": {"code": "action_conflict", "message": "The original action binds different inputs.", "receipt_id": "r1"}})
        return httpx.Response(200, json={"schema": "geocluster.science/1", "binding": {
            "projectId": "p", "scientificProjectId": "s", "bindingEpoch": "e"}, "intentId": "original-intent", "result": receipt})

    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    server = FastMCP("science")
    server.add_middleware(runctx.RunDirMiddleware())
    geomodel.register_tools(server)
    args = {"batch": {"operations": []}, "command_id": "c1", "idempotency_key": "stable-action",
            "run_dir": "runs/ses_root", "actingSessionId": "ses_child", "rootConversationId": "ses_root",
            "invocationId": "call-1"}
    # Use a registry-valid minimal batch, avoiding a test-owned engine model.
    batch_schema = geomodel.input_schema("apply")["definitions"]["OperationBatch"]
    if "ops" in batch_schema.get("properties", {}):
        args["batch"] = {"ops": []}
    async def journey():
        async with Client(server) as client:
            first = await client.call_tool("geomodel_apply", copy.deepcopy(args), raise_on_error=False)
            assert first.is_error
            assert "original" in first.content[0].text
            args["invocationId"] = "different-rpc"
            second = await client.call_tool("geomodel_apply", copy.deepcopy(args))
            assert second.structured_content["result"] == receipt
            assert second.structured_content["metadata"]["intentId"] == "original-intent"
            listed = {t.name: t for t in await client.list_tools()}
            Draft7Validator(listed["geomodel_apply"].outputSchema).validate(second.structured_content)
            args["command_id"] = "changed-command"
            conflict = await client.call_tool("geomodel_apply", copy.deepcopy(args), raise_on_error=False)
            assert conflict.is_error
            assert conflict.structured_content["error"]["code"] == "action_conflict"
            assert conflict.structured_content["error"]["receipt_id"] == "r1"
            assert conflict.structured_content["metadata"]["intentId"] == "original-intent"
    asyncio.run(journey())
    assert [c["actionId"] for c in calls] == ["stable-action"] * 3
    assert all("session_id" not in c["request"] for c in calls)
    assert calls[1]["trustedAttribution"]["actingSessionId"] == "ses_child"
    assert calls[0]["run_dir"] == "runs/ses_root"


def test_identity_overrides_rejected():
    from tools.geomodel import input_schema
    schema = input_schema("session")
    for key in ("token", "session_id", "principal", "trustedAttribution", "rootConversationId"):
        with pytest.raises(ValidationError):
            Draft7Validator(schema).validate({key: "forged"})


def test_compact_results_do_not_dump_artifact_bytes():
    from tools.geomodel import compact, without_credentials
    result = compact({"job_id": "j", "data_hex": "abcdef" * 1000, "values": list(range(1000))})
    assert "abcdef" not in json.dumps(result)
    assert len(json.dumps(result)) < 1000
    assert without_credentials({"job": {"session_id": "owner", "token": "never-output", "access_token": "secret"}}) == {"job": {"session_id": "owner"}}


def test_capabilities_use_private_get_without_engine_credentials(tmp_path, monkeypatch):
    from tools import geomodel
    monkeypatch.setattr(config, "WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(runctx, "SHARED", True)
    token = tmp_path / "control"
    token.write_text("private-control")
    monkeypatch.setenv("SCIENTIFIC_RUNTIME_URL", "http://127.0.0.1:3001/internal/science")
    monkeypatch.setenv("SCIENTIFIC_RUNTIME_TOKEN_FILE", str(token))
    def handler(request):
        assert request.method == "GET" and request.url.path == "/internal/science/capabilities"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"schema": "geocluster.science/1", "binding": None,
            "result": {"executionProfile": "development", "releaseQualified": False, "available": False}})
    monkeypatch.setattr(geomodel, "client_factory", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    server = FastMCP("science")
    server.add_middleware(runctx.RunDirMiddleware())
    geomodel.register_tools(server)
    async def journey():
        async with Client(server) as client:
            result = await client.call_tool("geomodel_session", {"run_dir": "runs/ses_root", "actingSessionId": "ses_child",
                "rootConversationId": "ses_root", "invocationId": "call-1"})
            assert result.structured_content["result"]["capabilities"]["releaseQualified"] is False
            assert result.structured_content["result"]["session"] is None
    asyncio.run(journey())
