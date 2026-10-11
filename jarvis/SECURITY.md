# Security policy

## Supported release line

Security fixes are applied to the latest `0.x` release until a stable `1.0`
policy is published.

## Reporting a vulnerability

Do not open a public issue for an exploitable vulnerability. Use GitHub's
private vulnerability reporting for `abdoo606/ai`. Include reproduction steps,
impact, affected version, and suggested mitigation when possible. Do not include
real credentials or private user data.

## Deployment baseline

- The packaged web command binds to `127.0.0.1` by default.
- If exposed beyond loopback, set a long random `JARVIS_ACCESS_TOKEN`, terminate
  TLS in a maintained reverse proxy, and restrict network access.
- Web fetching blocks loopback, private, link-local, reserved IPs, URL
  credentials, unsafe redirects, unsupported content types, and pages over 2 MB.
- Keep `JARVIS_ALLOW_PRIVATE_WEB=0` unless local-network reading is intentional.
- Command execution is absent unless both `JARVIS_ALLOW_SHELL=1` and a minimal
  `JARVIS_ALLOWED_COMMANDS` allowlist are set. Commands run without a shell.
- The model installer resolves exact Hub commit revisions and records them in
  `manifest.json`. Preserve this manifest in release evidence; never silently
  update production weights from a mutable branch.
- Model weights, memory databases, access tokens, and workspaces must never be
  committed or included in support bundles.
- Run as an unprivileged OS account. Do not run Jarvis as root.

## Threat-model limits

Local models can be prompt-injected by untrusted web pages. Tool boundaries,
source citations, workspace confinement, command allowlisting, and destructive
action confirmation reduce risk but do not make arbitrary autonomous execution
safe. Keep command execution disabled in consumer builds unless the product has
additional OS-level sandboxing and explicit per-action confirmation.
