# Zehnora closing and handover checklist

## Before signing

- [ ] Buyer and seller counsel approve Asset Purchase Agreement and schedules.
- [ ] Rights/title and prior MIT-distribution impact are expressly disclosed.
- [ ] Brand clearance and transferability are confirmed or brand is excluded.
- [ ] Third-party/model materials are identified as included, excluded, or
      buyer-downloaded under their own licenses.
- [ ] Purchase price, currency, tax allocation, escrow, holdback, and remedies
      are written in the definitive agreement.
- [ ] Transition support hours, response times, and end date are fixed.

## Closing deliverables

Build the reproducible technical archive with:

```bash
python -m zehnora.scripts.build_asset_bundle --output sale_artifacts --version 0.7.0
```

The output includes a ZIP, an external ZIP checksum, and an internal per-file
manifest. Rebuild it from the exact closing commit and compare hashes before use.

- [ ] Clean source bundle and SHA-256 manifest
- [ ] Exact closing commit ID and read-only history export if agreed
- [ ] CI/test/security scan reports
- [ ] Dependency SBOM and third-party notice schedule
- [ ] Model repository IDs, immutable revisions, and license archive
- [ ] Architecture, operations, security, and release documentation
- [ ] Repository/domain/package-name transfer instruments, if included
- [ ] Signing/update keys transferred through an approved secret channel—not Git
- [ ] Seller deletion/retention certificate as required by the agreement

## Buyer acceptance

- [ ] Buyer verifies bundle hashes.
- [ ] Buyer builds package in a clean environment.
- [ ] Buyer runs all tests and an agreed model demonstration.
- [ ] Buyer confirms administrative access transfer.
- [ ] Escrow agent releases funds only under the signed closing instructions.

## After closing

- [ ] Rotate all transferred credentials and signing keys.
- [ ] Remove seller access according to the agreement.
- [ ] Complete transition support and obtain written acceptance.
- [ ] Archive signed transaction records and tax evidence.
