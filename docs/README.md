# V-Gate Documentation

This directory holds long-form design proposals and validation reports. The [project README](../README.md) is the entry point and links to every document worth reading; the [online serving roadmap](../ROADMAP.md) holds the staged engineering plan for the serving plane.

## Status Conventions

Documents in this repository use these status labels:

- **Implemented**: present in the current codebase and covered by tests or runnable configuration.
- **Partial**: implemented for some backends or paths but not all; the gaps are stated explicitly.
- **Planned**: intended roadmap item without a complete implementation.
- **Design proposal**: exploratory architecture that should not be read as current runtime behavior.

When contributing documentation, keep implemented behavior separate from future plans. This makes the repository easier to evaluate and prevents roadmap items from being mistaken for shipped features.
