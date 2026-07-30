# V-Gate Documentation

This directory contains public engineering documentation for V-Gate.

## Core Documents

- [Project roadmap](../ROADMAP.md): staged engineering plan from the current single-node gateway toward reliable and distributed inference serving.
- [Advanced roadmap](design/ADVANCED_ROADMAP.md): superseded by [ROADMAP.md](../ROADMAP.md); kept for historical reference only.
- [V2 architecture proposal](design/V2_ARCHITECTURE_PROPOSAL.md): design proposal for a Python control plane with C++/CUDA data-plane components.
- [Containerization test report](reports/CONTAINERIZATION_TEST_REPORT.md): Docker and containerization validation notes.

## Status Conventions

Public design documents use these meanings:

- **Implemented**: present in the current codebase and covered by tests or runnable configuration.
- **In progress**: partially implemented or actively being integrated.
- **Planned**: intended roadmap item without a complete implementation.
- **Design proposal**: exploratory architecture that should not be read as current runtime behavior.

When contributing documentation, keep implemented behavior separate from future plans. This makes the repository easier to evaluate and prevents roadmap items from being mistaken for shipped features.
