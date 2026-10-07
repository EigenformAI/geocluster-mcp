"""Registry-derived scientific MCP facade over the runtime's private adapter.

The adapter owns binding authorization, durable logical-action mapping, source
capture/transfer and recorded job ownership. This process never opens a science
store, executes a scientific worker or receives an engine bearer credential.
"""
from __future__ import annotations

import copy
import json
import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import Tool, ToolResult
from jsonschema import Draft7Validator, ValidationError
from mcp.types import CallToolResult, ToolAnnotations

from . import config

OPERATIONS = (
    "session", "get", "explain", "query", "compare", "list_operations",
    "plan_recipe", "preview_operations", "apply", "ingest_sources",
    "submit_job", "poll_job", "job_result", "cancel_job", "fetch_artifact",
    "rank_hypothesis_set", "read_intent", "reconcile_intent", "prepare_review",
)
ENGINE_NAMES = {"compare": "compare_commits"}
MUTATIONS = frozenset({"apply", "ingest_sources", "submit_job", "prepare_review"})
CONTROLS = frozenset({"cancel_job", "reconcile_intent"})
HOST_FIELDS = frozenset({"principal", "principal_id", "session_id", "token", "session_token",
                         "bearer_token", "access_token", "engine_token"})
HOOK_FIELDS = ("actingSessionId", "rootConversationId", "invocationId")
CALL_ATTRIBUTION: ContextVar = ContextVar("scientific_call_attribution", default=None)
client_factory = httpx.AsyncClient
REGISTRY_PATH = Path(__file__).with_name("geomodel-registry.json")


@lru_cache(maxsize=1)
def engine_operations():
    return {op["name"]: op for op in json.loads(REGISTRY_PATH.read_text())["operations"]}


def without_host_identity(schema):
    """Remove authority only from engine request/Command nodes, not scientific evidence."""
    schema = copy.deepcopy(schema)
    for node in (schema, schema.get("definitions", {}).get("Command", {})):
        for key in HOST_FIELDS:
            node.get("properties", {}).pop(key, None)
        if "required" in node:
            node["required"] = [key for key in node["required"] if key not in HOST_FIELDS]
    return schema


ACTION_DESCRIPTION = (
    "Caller-selected stable logical action key. Reuse this exact key and unchanged request after response loss, "
    "restart or unknown admission; a changed payload conflicts. Never derive it from a tool/RPC invocation ID. "
    "Observe or reconcile the original intent; do not autonomously create a new retry after an unknown outcome. "
    "An explicit terminal retry/replay uses a new action key and retry_of/replay_of containing the original receipt ID."
)


def input_schema(operation):
    engine = engine_operations().get(ENGINE_NAMES.get(operation, operation))
    if engine:
        schema = without_host_identity(engine["request_schema"])
    elif operation in ("read_intent", "reconcile_intent"):
        schema = {"type": "object", "properties": {"intent_id": {"type": "string", "minLength": 1}},
                  "required": ["intent_id"]}
    else:
        # Integration selection is owner-validated, not a replacement engine model.
        schema = {"type": "object", "properties": {"selection": {"type": "object"},
                  "idempotency_key": {"type": "string"}}, "required": ["selection", "idempotency_key"]}
    schema["additionalProperties"] = False
    if operation in MUTATIONS:
        schema["properties"]["idempotency_key"].update(minLength=1, description=ACTION_DESCRIPTION)
    if operation == "ingest_sources":
        schema["description"] = ("Sources name workspace inputs for protected runtime capture. The adapter verifies "
                                 "complete bytes and supplies owner-staged paths; never supply a remote owner path.")
    return schema


COMPACT_SCHEMA = {"type": "object", "properties": {"kind": {"const": "compact"},
                  "summary": {"type": "object"}, "detail": {"type": "object"}},
                  "required": ["kind", "summary", "detail"], "additionalProperties": False}
INTENT_STATES = ["recorded", "uploading", "forwarding", "unknown", "preparing", "accepted", "ready", "failed", "abandoned", "transfer-failed"]
BINDING_SCHEMA = {"type": "object", "properties": {key: {"type": "string", "minLength": 1} for key in ("projectId", "scientificProjectId", "bindingEpoch")}, "required": ["projectId", "scientificProjectId", "bindingEpoch"], "additionalProperties": False}
OBSERVATION_SCHEMA = {"type": "object", "properties": {
    "kind": {"const": "intent_observation"}, "intentId": {"type": "string", "minLength": 1},
    "operation": {"enum": sorted(MUTATIONS)}, "state": {"enum": INTENT_STATES}, "binding": BINDING_SCHEMA,
    "result": {"type": ["object", "null"]}, "progress": {"type": "object"}},
    "required": ["kind", "intentId", "operation", "state", "binding", "result", "progress"], "additionalProperties": False}
JOB_OBSERVATION_VERSION = "geocluster.science.job-observation/1"


def embedded_response(operation, definitions):
    schema = copy.deepcopy(engine_operations()[operation]["response_schema"])
    # Each engine schema retains its own definitions when several operations share one output union.
    prefix = operation + "__"
    schema = json.loads(json.dumps(schema).replace('"#/definitions/', '"#/definitions/' + prefix))
    definitions.update({prefix + key: value for key, value in schema.pop("definitions", {}).items()})
    schema.pop("$schema", None)
    return schema


def job_observation_schema(definitions=None):
    """Host wrapper over exact registry JobRecord, JobReceipt and JobEvent schemas."""
    standalone = definitions is None
    if standalone:
        definitions = {}
    poll = embedded_response("poll_job", definitions)["properties"]
    cancel = embedded_response("cancel_job", definitions)["properties"]
    schema = {"type": "object", "properties": {
        "schema": {"const": JOB_OBSERVATION_VERSION}, "operation": {"enum": ["poll_job", "cancel_job"]},
        "job": copy.deepcopy(poll["job"]), "receipt": {"anyOf": [copy.deepcopy(cancel["receipt"]), {"type": "null"}]},
        "events": copy.deepcopy(poll["events"]), "eventsExhaustive": copy.deepcopy(poll["exhaustive"]),
        "nextEventCursor": copy.deepcopy(poll["next_event_cursor"]),
        "control": {"enum": ["available", "session-unavailable", "recovery-pending"]}, "readReceiptId": {"const": None}},
        "required": ["schema", "operation", "job", "receipt", "control", "readReceiptId"], "additionalProperties": False}
    if standalone:
        schema.update(definitions=definitions, **{"$schema": "http://json-schema.org/draft-07/schema#"})
    return schema


def output_schema(operation):
    definitions = {}
    name = ENGINE_NAMES.get(operation, operation)
    variants = [COMPACT_SCHEMA]
    if name in engine_operations():
        variants.append(embedded_response(name, definitions))
    if operation in {"poll_job", "cancel_job"}:
        variants.append(job_observation_schema(definitions))
    if operation in MUTATIONS | {"read_intent", "reconcile_intent"}:
        variants.append(OBSERVATION_SCHEMA)
    if operation in {"read_intent", "reconcile_intent"}:
        variants.extend(embedded_response(op, definitions) for op in sorted(MUTATIONS - {"prepare_review"}))
    if operation == "session":
        variants.append({"type": "object", "properties": {"kind": {"const": "session_facts"},
            "capabilities": {"type": "object"}, "session": {"anyOf": [embedded_response("session", definitions), {"type": "null"}]},
            "sessionAvailability": {"enum": ["available", "unavailable", "unsupported"]}, "sessionError": {"type": ["object", "null"]}},
            "required": ["kind", "capabilities", "session", "sessionAvailability", "sessionError"], "additionalProperties": False})
    result = {"anyOf": variants}
    return {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "definitions": definitions,
            "properties": {"result": result, "metadata": {"type": "object"}, "error": {"type": "object"}},
            "required": ["metadata"], "oneOf": [{"required": ["result"]}, {"required": ["error"]}],
            "additionalProperties": False}


def compact(value, depth=0):
    """Bound textual summaries without changing the schema-valid engine result."""
    if depth > 6:
        return {"omitted": True}
    if isinstance(value, dict):
        return {key: ({"omitted": "artifact bytes; use an explicit bounded fetch_artifact page or prepare_review"} if key in {"data_hex", "bytes", "data_base64"}
                      else compact(item, depth + 1)) for key, item in list(value.items())[:32] if key not in HOST_FIELDS}
    if isinstance(value, list):
        if len(value) > 16:
            return {"count": len(value), "omitted": True}
        return [compact(item, depth + 1) for item in value]
    if isinstance(value, str) and len(value) > 512:
        return {"characters": len(value), "omitted": True}
    return value


def needs_compaction(value):
    if isinstance(value, dict):
        return any(key in {"data_hex", "bytes", "data_base64"} or needs_compaction(item) for key, item in value.items())
    if isinstance(value, list):
        return len(value) > 16 or any(needs_compaction(item) for item in value)
    return isinstance(value, str) and len(value) > 512


def reject_authority(value):
    for node in (value, value.get("command", {})):
        if isinstance(node, dict) and (HOST_FIELDS & node.keys() or {"trustedAttribution", "attribution", *HOOK_FIELDS} & node.keys()):
            raise ToolError("geophysics geomodel: caller-selected authority is not accepted. Next step: use the scientific inputs only.")


def normalize_observation(value, envelope, operation, arguments, metadata):
    """An integration observation is accepted only with the original binding, intent and operation."""
    if not isinstance(value, dict) or not {"intentId", "operation", "state", "binding", "result"} <= value.keys():
        raise ValidationError("not an owner intent observation")
    expected = arguments.get("intent_id") if operation in {"read_intent", "reconcile_intent"} else envelope.get("intentId")
    if not expected or value["intentId"] != expected or value["intentId"] != envelope.get("intentId") or value["binding"] != envelope.get("binding"):
        raise ValidationError("observation does not match original intent/binding")
    original = value["operation"]
    if original not in MUTATIONS or operation not in {"read_intent", "reconcile_intent"} and original != operation:
        raise ValidationError("observation operation mismatch")
    allowed_states = ({"preparing", "ready", "failed", "unknown"} if original == "prepare_review" else
                      {"recorded", "uploading", "forwarding", "unknown", "accepted", "abandoned", "transfer-failed"} if original == "ingest_sources" else
                      {"recorded", "forwarding", "unknown", "accepted"})
    if value["state"] not in allowed_states:
        raise ValidationError("observation state does not belong to this operation")
    observation = {"kind": "intent_observation", **{key: value[key] for key in ("intentId", "operation", "state", "binding")},
        "result": None, "progress": {key: compact(value[key]) for key in ("control", "observedAt", "freshness", "payloadDigest", "execution", "attribution") if key in value}}
    Draft7Validator(OBSERVATION_SCHEMA).validate(observation)
    metadata["originalOperation"] = original
    if original in engine_operations() and value["state"] == "accepted":
        Draft7Validator(engine_operations()[original]["response_schema"]).validate(value["result"])
        metadata["observation"] = observation
        return value["result"]
    if original == "prepare_review" and value["state"] in {"ready", "failed"}:
        if not isinstance(value["result"], dict):
            raise ValidationError("review outcome missing")
        observation["result"] = compact(value["result"])
    elif value["result"] is not None:
        raise ValidationError("pending observation cannot claim a settled result")
    return observation


def normalize_job_observation(value, operation, arguments):
    Draft7Validator(job_observation_schema()).validate(value)
    job, receipt = value["job"], value["receipt"]
    if operation not in {"poll_job", "cancel_job"} or value["operation"] != operation or job["job_id"] != arguments.get("job_id"):
        raise ValidationError("supervisory observation does not select the requested original job/operation")
    if receipt is not None:
        if job["status"] not in {"succeeded", "failed", "cancelled", "interrupted"} or receipt["job_id"] != job["job_id"] or receipt["status"] != job["status"]:
            raise ValidationError("terminal receipt does not match the observed original job")
        if job.get("receipt_id") is not None and job["receipt_id"] != receipt["receipt_id"]:
            raise ValidationError("terminal receipt identity does not match the observed original job")
    # Preserve the host outcome, including failed/not-started settlement, and null read-receipt identity.
    return value


async def session_facts(client, url, headers, body, envelope):
    capabilities = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    operations = capabilities.get("operations", [])
    supports = any(op == "session" or isinstance(op, dict) and op.get("name") == "session" for op in operations)
    value = {"kind": "session_facts", "capabilities": compact(capabilities), "session": None,
             "sessionAvailability": "unsupported" if not supports and capabilities.get("available", True) and not envelope.get("error") else "unavailable", "sessionError": envelope.get("error")}
    if supports and capabilities.get("available", True) and not envelope.get("error"):
        try:
            response = await client.post(url + "/call", headers=headers, json=body, timeout=30)
            session = without_credentials(response.json())
            if session.get("schema") != "geocluster.science/1" or session.get("binding") != envelope.get("binding"):
                raise ValidationError("session binding mismatch")
            if response.is_error or session.get("error"):
                value["sessionError"] = session.get("error") or {"code": "session_unavailable", "message": "Host-managed session facts are unavailable."}
            else:
                Draft7Validator(engine_operations()["session"]["response_schema"]).validate(session.get("result"))
                value.update(session=session["result"], sessionAvailability="available")
        except (httpx.HTTPError, ValueError, ValidationError, AttributeError):
            value["sessionError"] = {"code": "session_unavailable", "message": "Host-managed session facts are unavailable."}
    return {"schema": "geocluster.science/1", "binding": envelope.get("binding"), "result": value}


def without_credentials(value):
    if isinstance(value, dict):
        return {key: without_credentials(item) for key, item in value.items()
                if key not in HOST_FIELDS - {"principal", "principal_id", "session_id"}
                and key.lower() not in {"authorization", "credentials", "bearer", "secret"}}
    if isinstance(value, list):
        return [without_credentials(item) for item in value]
    return value


def runtime_connection():
    url = os.environ.get("SCIENTIFIC_RUNTIME_URL", "").rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/internal/science" or parsed.query or parsed.fragment or parsed.username:
        raise ToolError("geophysics geomodel: private scientific adapter is unavailable. Next step: configure SCIENTIFIC_RUNTIME_URL on the host.")
    try:
        token = Path(os.environ["SCIENTIFIC_RUNTIME_TOKEN_FILE"]).read_text().strip()
        if not token:
            raise ValueError("empty credential")
    except (KeyError, OSError, ValueError) as exc:
        raise ToolError("geophysics geomodel: runtime control credential is unavailable. Next step: restart the runtime supervisor.") from exc
    return url, {"x-gc-runtime-control": token}


class ScienceErrorResult(ToolResult):
    def to_mcp_result(self):
        return CallToolResult(content=self.content, structuredContent=self.structured_content, isError=True)


class GeomodelTool(Tool):
    operation: str

    async def run(self, arguments):
        try:
            Draft7Validator(self.parameters).validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"geophysics {self.name}: invalid scientific request at {list(exc.path)}. Next step: correct the tool inputs.") from exc
        reject_authority(arguments)
        if self.operation == "ingest_sources":
            try:
                for source in arguments["sources"]:
                    config.resolve_path(source["path"])
            except ValueError as exc:
                raise ToolError(f"geophysics {self.name}: source path is outside the workspace. Next step: select a captured workspace source.") from exc
        run = config.current_run_dir()
        attribution = CALL_ATTRIBUTION.get()
        if not run or not attribution:
            raise ToolError(f"geophysics {self.name}: trusted conversation scope is missing. Next step: restart the runtime; do not retry a mutation with a new action key.")
        relative = os.path.relpath(run, config.WORKSPACE_ROOT).replace(os.sep, "/")
        config.validate_call_run_dir(relative)
        if attribution["rootConversationId"] != relative.split("/")[1]:
            raise ToolError(f"geophysics {self.name}: conversation scope disagrees. Next step: restart the runtime scope hook.")
        url, headers = runtime_connection()
        operation = ENGINE_NAMES.get(self.operation, self.operation)
        body = {"operation": operation, "request": copy.deepcopy(arguments), "run_dir": relative,
                "actingSessionId": attribution["actingSessionId"], "trustedAttribution": attribution}
        if self.operation in MUTATIONS:
            body["actionId"] = arguments["idempotency_key"]
        if self.operation in {"read_intent", "reconcile_intent"}:
            body["intentId"] = arguments["intent_id"]
        if len(json.dumps(body).encode()) > 256 * 1024:
            raise ToolError(f"geophysics {self.name}: request exceeds 256 KiB. Next step: reduce this scientific mutation.")
        try:
            async with client_factory() as client:
                # No automatic retries: the runtime resolves the original action before any forwarding.
                if self.operation == "session":
                    response = await client.get(url + "/capabilities", headers=headers, timeout=30)
                else:
                    response = await client.post(url + "/call", headers=headers, json=body, timeout=30)
                envelope = without_credentials(response.json())
                if self.operation == "session" and isinstance(envelope, dict) and envelope.get("schema") == "geocluster.science/1":
                    envelope = await session_facts(client, url, headers, body, envelope)
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"geophysics {self.name}: status unknown; the original action may have been accepted. "
                            "Next step: observe/reconcile the original intent or repeat the identical original action key; never mint a replacement key.") from exc
        if not isinstance(envelope, dict) or envelope.get("schema") != "geocluster.science/1":
            raise ToolError(f"geophysics {self.name}: status unknown; invalid runtime response. Next step: reconcile the original action.")
        metadata = {key: envelope[key] for key in ("schema", "binding", "intentId") if key in envelope}
        metadata.update(operation=operation, attribution=attribution)
        if "actionId" in body:
            metadata["actionId"] = body["actionId"]
        if "error" in envelope:
            error = {key: envelope["error"][key] for key in ("code", "message", "receipt_id", "dispatch", "nextAction") if key in envelope["error"]}
            error.setdefault("nextAction", "Inspect or reconcile the original intent; preserve its action key.")
            if "result" in envelope:
                metadata["observation"] = compact(envelope["result"])
            result = ScienceErrorResult(content=json.dumps(compact({"error": error, "metadata": metadata})),
                                        structured_content={"error": error, "metadata": metadata})
        else:
            if response.is_error and self.operation != "session" or "result" not in envelope:
                raise ToolError(f"geophysics {self.name}: status unknown. Next step: reconcile the original action.")
            value = envelope["result"]
            engine = engine_operations().get(operation)
            if self.operation != "session":
                try:
                    if isinstance(value, dict) and value.get("schema") == JOB_OBSERVATION_VERSION:
                        value = normalize_job_observation(value, self.operation, arguments)
                    elif self.operation in {"read_intent", "reconcile_intent", "prepare_review"} or isinstance(value, dict) and "intentId" in value:
                        value = normalize_observation(value, envelope, self.operation, arguments, metadata)
                    elif engine:
                        Draft7Validator(engine["response_schema"]).validate(value)
                except ValidationError as exc:
                    raise ToolError(f"geophysics {self.name}: runtime result does not match the engine registry. "
                                    "Next step: inspect the original intent; do not resubmit with a new key.") from exc
            summary = compact(value)
            # An explicit artifact fetch is a real engine-bounded byte page, not an invented product URL.
            if self.operation not in {"session", "fetch_artifact"} and value.get("kind") != "intent_observation" and (needs_compaction(value) or len(json.dumps(value)) > 12_000):
                value = {"kind": "compact", "summary": summary, "detail": {
                    "operation": operation, "lookup": compact(arguments), "intentId": metadata.get("intentId"),
                    "nextAction": "Use a bounded fetch_artifact page; prepare_review is required before registered review products exist."}}
            structured = {"result": value, "metadata": metadata}
            result = ToolResult(content=json.dumps(compact(structured)), structured_content=structured)
        Draft7Validator(self.output_schema).validate(result.structured_content)
        return result


def register_tools(server):
    for operation in OPERATIONS:
        description = f"Scientific {operation}: delegate to the project owner through the runtime adapter."
        if operation in MUTATIONS:
            description += " " + ACTION_DESCRIPTION
        if operation == "cancel_job":
            description += " Cancel only a recorded job through its original owner scope. A request is not proof of cancellation; preserve the actual terminal outcome."
        if operation == "session":
            description += " Inspect capabilities/budgets of the host-managed session; never mint or expose credentials."
        if operation == "fetch_artifact":
            description += " Explicitly read an engine-bounded original byte page in structuredContent. Use a small max_bytes; routine text omits bytes. Products exist only after prepare_review."
        server.add_tool(GeomodelTool(name=f"geomodel_{operation}", operation=operation,
            description=description, parameters=input_schema(operation), output_schema=output_schema(operation),
            annotations=ToolAnnotations(readOnlyHint=operation not in MUTATIONS | CONTROLS,
                                        destructiveHint=operation in MUTATIONS | CONTROLS, openWorldHint=False)))
