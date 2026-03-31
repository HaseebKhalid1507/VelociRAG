# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.7.x   | ✅ |
| < 0.7   | ❌ |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public issue.**

Instead, report it privately:
1. Go to the [Security tab](https://github.com/HaseebKhalid1507/VelociRAG/security/advisories)
2. Click "Report a vulnerability"
3. Provide a description, steps to reproduce, and potential impact

You should receive a response within 72 hours. We'll work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

VelociRAG runs locally and processes local files. Security considerations include:
- **SQL injection** in search queries (mitigated via parameterized queries)
- **Path traversal** in file indexing
- **FTS5 query injection** (mitigated via input sanitization)
- **ONNX model integrity** (models downloaded from HuggingFace Hub)

## Disclosure Policy

We follow coordinated disclosure. Once a fix is available, we'll publish a security advisory with credit to the reporter.
