# Security Policy

## Supported versions

`ai-sdk-cpp` is pre-1.0. Security fixes target the latest `main`.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems. Instead,
email the maintainers directly with a description and, if possible, a
reproduction. We will acknowledge within a reasonable window and coordinate a
fix and disclosure.

While we work on a fix, please give us time to remediate before publishing
details.

## Threat model notes

`ai-sdk-cpp` executes **arbitrary shell commands and filesystem writes** when an
agent is given the standard toolkit (`bash`, `write_file`, `edit_file`, …) —
exactly like Claude Code / Codex. Applications embedding the SDK are responsible
for:

- gating dangerous tools with the **permission/approval hooks**
  (`ai::with_permissions`) before handing control to untrusted prompts;
- sandboxing the working directory and process;
- treating all model output as untrusted (it can contain malicious tool calls).

API keys are read from environment variables or passed explicitly; never commit
keys. Live/integration tests are gated behind env vars so secrets are not
required to run the offline suite.
