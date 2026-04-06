# Security Policy

## Supported Versions
| Version | Supported | Notes |
|---|---|---|
| latest | Yes | Only the latest PyPI release receives security patches and fixes. |

## Reporting a Vulnerability
If you discover a security issue, please report it via email:

- security@kanx-project.org

We commit to an initial response within 48 hours and a disclosure timeline of 90 days.

## What Is Not In Scope
The following are outside the scope of this repository's security process:

- Benchmark accuracy disputes or claims about model performance.
- Vulnerabilities in third-party dependencies not maintained in this repository.
- Issues with external deployment infrastructure beyond the provided manifests.

## FastAPI Deployment Guidance
The FastAPI serving layer must never be exposed directly to the public internet without:

- an authentication layer,
- TLS termination,
- and appropriate network access controls.
