# V-Gate Documentation

This directory holds validation reports and the documentation conventions used across the project. The [project README](../README.md) is the entry point and the [roadmap](../ROADMAP.md) covers planned work.

## Reports

- [Containerization test report](reports/CONTAINERIZATION_TEST_REPORT.md): Docker build and runtime validation notes.

## Status Conventions

Documents in this repository use these status labels:

- **Implemented**: present in the current codebase and covered by tests or runnable configuration.
- **Partial**: implemented for some backends or paths but not all; the gaps are stated explicitly.
- **Planned**: intended roadmap item without a complete implementation.

When contributing documentation, keep implemented behavior separate from future plans. This makes the repository easier to evaluate and prevents roadmap items from being mistaken for shipped features.
