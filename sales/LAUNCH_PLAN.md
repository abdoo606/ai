# Zehnora commercial launch plan

## Gate 0 — brand and ownership

- Complete the searches in `BRAND_CLEARANCE.md`.
- Confirm ownership/contributor rights for all new code.
- Review the imported MIT license and all dependency/model notices.
- Decide whether to keep MIT, dual-license future code, or sell support/builds.

## Gate 1 — product evidence

- Benchmark Arabic and English quality on each advertised model profile.
- Publish RAM/VRAM, disk, first-token latency, throughput, and task-success data.
- Run prompt-injection, citation, malformed-input, and destructive-action suites.
- Commission an independent penetration test and close critical/high findings.
- Add OS sandboxing or keep command execution disabled in retail builds.

## Gate 2 — commercial operations

- Form/select the selling legal entity and bank account.
- Obtain tax/VAT advice for target markets.
- Draft counsel-reviewed Terms, Privacy Notice, EULA/license notices, refund
  policy, support SLA, acceptable-use policy, and data-processing terms.
- Select a payment provider and invoicing flow; never store card data directly.
- Select support email/help desk and define supported versions.

## Gate 3 — distributable product

- Build signed Windows/macOS/Linux installers for chosen launch platforms.
- Publish checksums, SBOM, model manifest, release notes, and update policy.
- Test clean install, update, rollback, uninstall, export, and deletion.
- Keep model downloads resumable and pinned to recorded immutable revisions.

## Gate 4 — paid beta

- Recruit 3–5 design partners under a written beta agreement.
- Charge only after brand/legal/payment gates are approved.
- Track activation, task success, crashes, support burden, refund reasons, and
  retention without enabling telemetry unless the customer explicitly opts in.
- Hold a launch review before broader advertising.

## Current status

Engineering commercial-beta baseline: implemented. CI, packaging, security
controls, vulnerability scans, and release checklist: implemented. Brand search,
legal documents, payment account, signed native installers, real-model benchmark,
and independent penetration test: externally blocked/pending.
