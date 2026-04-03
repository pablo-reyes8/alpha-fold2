# Security Policy

## Supported Versions

This repository is a research prototype, not a hardened production service. Security support is currently limited to the latest state of the default branch and the repository-maintained configs and scripts.

| Version | Supported |
| --- | --- |
| Latest default branch | Yes |
| Older commits or tags | Best effort only |
| Personal forks | No |
| Modified notebook copies outside this repository | No |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Instead:

1. Use GitHub private vulnerability reporting for this repository if it is enabled.
2. If private reporting is not enabled, contact the repository maintainer privately before any public disclosure.
3. Include enough detail to reproduce and triage the issue safely.

## What to Include

Please include, when possible:

- A short description of the vulnerability and the impacted area
- Affected files, scripts, or commands
- Reproduction steps
- Any proof-of-concept input needed to trigger the issue
- Security impact and likely attack surface
- Whether secrets, credentials, or arbitrary file access are involved
- Suggested mitigations, if you have them

## Scope

The most relevant security-sensitive areas in this repository are:

- Data download and local preprocessing scripts
- Parsing of external structure or MSA files
- Training and validation CLIs that execute local workflows
- Dependency or environment configuration that could affect code execution
- Multi-GPU launch paths and distributed execution helpers

Examples of issues that may be security-relevant:

- Arbitrary command execution
- Path traversal or unintended file overwrite
- Unsafe handling of untrusted input files
- Credential leakage in configs, notebooks, logs, or scripts
- Dependency or supply-chain risks introduced by the repository setup

Examples that are usually not security issues by themselves:

- Model quality limitations
- Numerical instability without a security impact
- GPU out-of-memory failures
- Incorrect predictions or training divergence
- General performance regressions

## Disclosure Process

After receiving a private report, the maintainer will aim to:

- Acknowledge receipt within a reasonable timeframe
- Reproduce and assess the issue
- Work on a fix or mitigation
- Coordinate disclosure after a patch is available, when appropriate

Response times may vary because this is a research repository maintained alongside active experimentation, but private reports will be treated seriously.
