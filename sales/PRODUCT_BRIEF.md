# Zehnora product brief

## Positioning

Zehnora is a private, API-free Arabic/English personal research assistant that
runs on hardware controlled by the customer. It can research a field, archive
source-grounded knowledge, connect concepts, and use deliberate local model
passes without sending conversations to a hosted AI provider.

## Ideal first customers

- Arabic/English researchers and technical professionals with capable hardware
- Privacy-sensitive teams that cannot send internal prompts to hosted models
- Small organizations wanting a self-hosted research assistant

## Differentiators

- Local inference; no hosted-model account or AI API key
- Arabic/Levantine intent handling plus English responses
- Deep topic research with citations and resumable local memory
- Hybrid BM25 + concept-graph retrieval
- Intent → reasoner → critic → final-writer pipeline
- Hardware profiles from 4B through server-class 235B-A22B
- No telemetry by default
- Security controls, reproducible model revisions, and documented release gates

## Honest limitations

- Local quality and speed depend heavily on hardware and model profile.
- No profile should be marketed as “equal to Claude” without independent,
  reproducible benchmarks.
- Web research requires internet access and source sites can fail or rate-limit.
- The current deliverable is a commercial beta, not a certified high-risk or
  safety-critical system.

## Proposed beta offer

A self-hosted early-access package containing the signed application build,
hardware sizing help, onboarding, model installation, updates during the beta,
and defined support response times. Command execution remains disabled in the
consumer package unless an OS sandbox is completed.
