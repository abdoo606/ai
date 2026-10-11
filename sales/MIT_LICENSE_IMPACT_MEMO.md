# Historical MIT license impact memo

**Date:** 2026-10-11  
**Status:** diligence summary, not a legal opinion

## Executive conclusion

A sale remains possible, but it should not be marketed as a transfer of perfectly
exclusive source code. The core assistant was committed and publicly available
while a repository-level MIT license was present. The definitive agreement must
transfer only the seller's assignable rights and expressly preserve/disclose any
rights already granted to prior recipients.

## Repository facts

- The repository is public and currently reports zero stars and zero forks.
  Those counts do not prove that nobody viewed, downloaded, or cloned it.
- Assistant development appears in commits from `d4b4bbc` through `4d63742`
  under the earlier Jarvis path, followed by the Zehnora rename in `47862b7`.
- The repository-level MIT text remained in place until commit `4913ae0`.
- The former notice named Philip Shiu and Nico Spiller, originally associated
  with the unrelated connectome project. It did not separately identify the
  copyright owner of newly added assistant code. That creates a scope/title
  question for counsel rather than a safe basis for an exclusivity claim.
- The current proprietary evaluation notice started at `4913ae0`; it explicitly
  says it does not revoke prior grants.
- Commit metadata shows one seller identity using two email forms. Legal identity,
  authority, human authorship, AI-assisted-development terms, and any contributor
  or employment obligations still need documentary verification.

## What MIT says

The historical text grants a person obtaining a covered copy broad permission to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies,
subject to preserving the copyright and permission notice. GitHub's
ChooseALicense project describes MIT as permitting commercial use, modification,
distribution, and private use, and says modifications/larger works may be
released under different terms.

Primary records used here:

- Historical repository license: `git show 47862b7^:LICENSE`
- Current notice: [`../LICENSE`](../LICENSE)
- GitHub ChooseALicense MIT record:
  <https://github.com/github/choosealicense.com/blob/gh-pages/_licenses/mit.txt>

## Practical effect on a sale

### A seller can propose transferring

- copyright/title the seller actually owns in the current code;
- current proprietary additions and documentation, subject to authorship review;
- repository administration, product know-how, tests, CI, and transition support;
- assignable brand/domain/package rights expressly listed in the agreement.

### A buyer should not be promised

- cancellation of previously granted MIT rights;
- the ability to stop lawful users of an earlier MIT-covered copy;
- that no clone or private copy exists merely because GitHub shows zero forks;
- ownership of dependencies, model weights, fetched content, or unrelated history;
- trademark clearance or uncontested copyright title without professional review.

## Risk rating for pricing

| Issue | Current rating | Effect |
|---|---|---|
| Prior public MIT exposure | High | Materially reduces exclusivity premium |
| Ambiguous old copyright notice/scope | High | Counsel/title evidence required |
| Public repository remains visible | High | Continued exposure; changing visibility is not retroactive |
| Single apparent contributor identity | Medium | Easier chain, but identity/authority still unverified |
| Third-party packages/model weights | Medium | Must remain licensed/excluded, not assigned as owned |
| No known stars/forks | Helpful but weak | Not evidence that no copies exist |

## Required transaction language for counsel

The definitive agreement should define the exact assigned IP, list prior public
revisions and MIT exposure in a disclosure schedule, exclude third-party assets,
disclaim perfect exclusivity, allocate infringement/title risk, state acceptance
criteria, and control repository/credential transfer through escrow. Seller and
buyer counsel—not this memo—must determine whether the historical MIT file legally
covered each assistant revision and what representations are supportable.
