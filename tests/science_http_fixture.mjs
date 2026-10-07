// Actual runtime HTTP/adapter/store with a scripted owner: no scientific workers.
import http from "node:http"
import path from "node:path"
import { ScienceHttp } from "../../runtime/lib/science-http.js"
import { ScienceAdapter } from "../../runtime/lib/science.js"

const binding = { projectId: "p", scientificProjectId: "s", bindingEpoch: "e" }
const envelope = (result, intentId, error) => ({ schema: "geocluster.science/1", binding, ...(intentId ? { intentId } : {}), result, ...(error ? { error } : {}) })
const receipt = { command_id: "c1", diagnostics: [], payload_digest: "digest", receipt_id: "r1", recorded_at: "now", status: "applied", validations_passed: [], written_revisions: [] }
const job = { job_id: "job-original", principal: "owner", session_id: "owner-session", request_digest: "digest", status: "queued", submitted_at: "now" }
const observation = (id, operation, state, result = null) => ({ intentId: id, operation, state, binding, payloadDigest: "digest", principal: "owner", sessionId: "owner-session", attribution: { kind: "agent", rootConversationId: "ses_root", actingSessionId: "ses_child" }, result, control: "available", observedAt: 1, freshness: "current" })
let firstIntent
const owner = { async json(method, route, body) {
	if (route === "/science/capabilities") return envelope({ available: true, operations: ["session"], executionProfile: "development" })
	if (route === "/science/call") {
		if (body.operation === "apply") { firstIntent = body.intentId; return envelope(observation(firstIntent, "apply", "unknown"), firstIntent) }
		if (body.operation === "poll_job") return envelope({ freshness: "unavailable" }, firstIntent, { code: "session.original-unavailable", message: "The original execution session expired." })
	}
	const id = route.split("/").at(-1)
	if (id === "submission-intent") return envelope({ ...observation(id, "submit_job", "accepted", job), execution: { job } }, id)
	if (id === "wrong-intent") return envelope(observation("different-intent", "submit_job", "accepted", job), id)
	return envelope(observation(id, "apply", "accepted", receipt), id)
} }
const directory = process.argv[2]
const adapter = await ScienceAdapter.open({ config: { projectId: "p", workspaceDir: directory, scienceBinding: binding, scienceOwnerUrl: "http://scripted-owner", scienceStateDir: path.join(directory, "state") }, client: owner })
const router = new ScienceHttp(adapter)
const server = http.createServer((req, res) => {
	if (req.headers["x-gc-runtime-control"] !== "private-control") { res.writeHead(403); return res.end() }
	void router.handle(req, res, new URL(req.url, "http://localhost"), "internal")
})
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve))
console.log(JSON.stringify({ url: `http://127.0.0.1:${server.address().port}/internal/science` }))
process.on("SIGTERM", () => { server.close(async () => { await adapter.close(); process.exit(0) }); server.closeAllConnections() })
