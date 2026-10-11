# Zehnora technical transfer checklist

This repository is a production-oriented beta and not a legal, security, or
commercial certification. Complete every applicable gate before a one-time
whole-asset closing or buyer acceptance. Use the transaction room under
`../sales/` for title, contract, and handover work.

## Engineering gates

- [ ] CI passes on Python 3.10, 3.11, and 3.12.
- [ ] Unit, API-boundary, security, migration, packaging, and source-bundle tests pass.
- [ ] Install the wheel in a clean machine/VM and run CLI + web smoke tests.
- [ ] Build the deterministic clean-source archive; record its SHA-256 digest.
- [ ] Run real-model evaluations on every represented hardware profile.
- [ ] Measure time-to-first-token, tokens/second, peak RAM/VRAM, disk usage, and
      failure behavior for Arabic and English.
- [ ] Test interrupted downloads, corrupt weights, full disk, no internet,
      search rate limits, malformed pages, and database recovery.
- [ ] Add OS-level sandboxing before representing command execution as deployable.
- [ ] Produce a dependency lock/SBOM and perform vulnerability scans.
- [ ] Commission an independent penetration test before internet exposure.

## AI quality gates

- [ ] Maintain versioned Arabic/English intent and tool-use evaluation sets.
- [ ] Test incomplete text, dialect, misspellings, prompt injection, harmful
      instructions, unsupported claims, citation accuracy, and contradictions.
- [ ] Record model/profile-specific limitations; never represent equivalence to
      a frontier hosted model without reproducible independent benchmarks.
- [ ] Verify every model license for the buyer's intended territory and use.

## Transfer, privacy, and legal gates

- [ ] Counsel verifies seller identity, title, contributor rights, prior public
      revisions, historical MIT grants, and the definitive assignment perimeter.
- [ ] Review copyright/database rights for archived web content. Transfer of the
      software does not grant redistribution rights to learned source material.
- [ ] Complete professional brand, trademark, domain, and common-law clearance.
- [ ] Provide attribution and license notices for packages, models, and tooling.
- [ ] Confirm secrets and personal/buyer data are absent from source and history,
      or document a counsel-approved remediation plan.
- [ ] Agree which accounts, domains, repositories, signing keys, documentation,
      warranties, support period, and excluded assets transfer at closing.
- [ ] Obtain tax/accounting advice and execute definitive documents before funds
      and control are exchanged.

## Acceptance evidence

Place CI logs, test reports, archive/wheel hashes, SBOM, model manifests,
benchmarks, penetration-test disposition, disclosure schedules, executed closing
documents, and a countersigned acceptance record in the controlled data room.
Do not commit confidential transaction records, credentials, or model weights.
