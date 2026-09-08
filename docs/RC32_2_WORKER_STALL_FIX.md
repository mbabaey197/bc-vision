# RC32.2 worker stall fix

The frame commit held the camera model-switch lock while the baseline shadow observer requested the worker lock. Submission/status use the reverse order when inspecting persistence retries. The cycle exists even with shadow mode disabled because the observer first reads its setting.

The observer now runs after releasing the camera lock. Its optional failure cannot skip worker completion bookkeeping. OCR confidence and voting guards are unchanged. The bounded two-thread regression fails on the previous code and passes with this change.

Persistent diagnostics write `BCVision-runtime.log` in the configured data directory, rotating at 2 MiB with two backups. A daemon watchdog records Python thread stacks when an ANPR operation exceeds 30 seconds or the event-loop heartbeat exceeds 15 seconds, at most once per minute. No frame images or local-variable values are included. Processing errors and Uvicorn errors are retained. Camera startup/shutdown run outside the event loop.

The supplied startup log did not contain a deadlock stack. This fixes a reproduced defect consistent with the reported symptoms; it does not prove that all camera problems are resolved or establish real-world OCR accuracy.

Runtime ABI remains 2 and the immutable runtime base remains RC32. The standard transactional update installer stops the service before activation and preserves configuration, database and media. Interrupted video passes deliberately remain preview-only to prevent duplicate events. After updating, disable the interrupted test source and upload the video as a new test to start a fresh processing pass.
