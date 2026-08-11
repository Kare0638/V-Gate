# V-Gate Documentation

This directory holds long-form design proposals and validation reports. The [project README](../README.md) is the entry point; the [online serving roadmap](../ROADMAP.md) covers the serving track, and the [batch compute plane design](design/BATCH_PLANE.md) covers the planned Ray/Daft multimedia track.

## Design Documents

- [Batch compute plane](design/BATCH_PLANE.md): planned multimedia job architecture, execution boundary, ordered tasks, risks, and acceptance criteria.
- [V2 architecture proposal](design/V2_ARCHITECTURE_PROPOSAL.md): proposed C++/CUDA data-plane direction; not current runtime code.

## Status Conventions

Documents in this repository use these status labels:

- **Implemented**: present in the current codebase and covered by tests or runnable configuration.
- **Partial**: implemented for some backends or paths but not all; the gaps are stated explicitly.
- **Planned**: intended roadmap item without a complete implementation.
- **Design proposal**: exploratory architecture that should not be read as current runtime behavior.

When contributing documentation, keep implemented behavior separate from future plans. This makes the repository easier to evaluate and prevents roadmap items from being mistaken for shipped features.
