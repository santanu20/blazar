# W12 engine-surface probe — sd-server master-929-3f8527a (2026-10-06, rig :11501, CPU backend via BLAZAR_SDCPP_EXTRA_ARGS)

Mid-render surface (vid_gen job `generating`, polled 3x over 34s on child :39101):
- GET /sdcpp/v1/jobs/{id} → 200 {id, kind, status:"generating", queue_position:0, created, started, completed:null, error:null, result:null}
  → NO progress fraction, NO preview/partial-frame bytes. Field set is terminal-only.
- GET /sdapi/v1/progress → 404 (0B); with ?skip_current_image=false → 404 (0B)
- GET /health → 404 (0B)
- GET /sdcpp/v1/capabilities → 200 (control probe: child healthy during the above)

VERDICT: probed-negative — this engine build exposes no mid-render preview/progress surface.
`partial_images` and mid-render previews are unimplementable against it; Sora `progress`
serves 0 (queued) / 100 (completed) only, which matches the gateway mapping (omitted mid-run).

# Crash-window close-out live pins (same rig session)
1. Admission-evict orphan (image row job_6ac40234): lane GET /v1/images/jobs/{id} after the
   owning child was evicted → 200 {"status":"failed","error":"job's engine child is gone
   (eviction, crash or restart) — resubmit the generation"} (was: 404 "no live diffusion child").
2. SIGKILL orphan (video row job_6ac40235, kill -9 child mid-generating): GET /v1/videos/{id}
   → 200 {"status":"failed","error":"job's engine child is gone (eviction, crash or restart) —
   resubmit the generation"} (was: forever in_progress).
3. Terminal row + no live child (job_6ac40072 post boot-sweep): lane GET → 200 row_payload
   (was: 404 "no live diffusion child serves jobs — POST ... boots one").
4. W11 argv receipt: child argv tail `--backend cpu` from env BLAZAR_SDCPP_EXTRA_ARGS=--backend,cpu
   (/proc/905254/cmdline) — single occurrence, auto device-split suppressed.
