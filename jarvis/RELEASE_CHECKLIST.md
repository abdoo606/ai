# Commercial release checklist

This repository is a production-oriented beta, not a legal certification. A
seller should complete every applicable item before distributing a build.

## Engineering gates

- [ ] CI passes on Python 3.10, 3.11, and 3.12.
- [ ] Unit, API-boundary, security, migration, and packaging tests pass.
- [ ] Install the wheel in a clean machine/VM and run CLI + web smoke tests.
- [ ] Run real-model evaluations on every advertised hardware profile.
- [ ] Measure time-to-first-token, tokens/second, peak RAM/VRAM, disk usage, and
      failure behavior for Arabic and English.
- [ ] Test interrupted downloads, corrupt weights, full disk, no internet,
      search rate limits, malformed pages, and database recovery.
- [ ] Add OS-level sandboxing before enabling commands in a consumer build.
- [ ] Perform dependency/SBOM and vulnerability scans in the release pipeline.
- [ ] Commission an independent penetration test before internet exposure.

## AI quality gates

- [ ] Maintain versioned Arabic/English intent and tool-use evaluation sets.
- [ ] Test incomplete text, dialect, misspellings, prompt injection, harmful
      instructions, unsupported claims, citation accuracy, and contradictions.
- [ ] Publish model/profile-specific limitations; never advertise equivalence to
      a frontier hosted model without reproducible third-party benchmarks.
- [ ] Verify all model licenses for the intended commercial territory/use.

## Product, privacy, and legal gates

- [ ] Provide terms of service, privacy notice, refund/support policy, and data
      deletion/export controls appropriate to the markets where it is sold.
- [ ] Review copyright/database rights for archived web content. Selling the
      software does not grant redistribution rights to learned source material.
- [ ] Verify the product name, branding, and “Jarvis” trademark risk with counsel.
- [ ] Provide attribution and license notices for Python packages and models.
- [ ] Sign installers, publish checksums, define an update channel, and document
      vulnerability response and supported versions.
- [ ] Make telemetry opt-in; this code sends no telemetry by default.

## Release evidence

Attach CI logs, test report, wheel hashes, dependency lock/SBOM, model manifests,
benchmark results, penetration-test disposition, and signed legal approvals to
each commercial release record.
