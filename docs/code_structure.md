# Code Structure

Cosmos-H-Surgical is a focused package layered on a pinned public Cosmos
Framework revision. The full framework source is not copied into this
repository.

## Repository Layout

```text
Cosmos-H-Surgical/
|-- cosmos_h_surgical/
|   |-- cli.py
|   |-- checkpoints.py
|   |-- inference.py
|   |-- training.py
|   |-- provenance.py
|   |-- release.py
|   `-- configs/
|-- docs/
|-- examples/
|   |-- inference/
|   `-- post_training/
|-- tests/
|-- pyproject.toml
|-- uv.lock
|-- UPSTREAM.md
`-- release-manifest.json
```

The historical Cosmos 2.5 implementation remains available on the
`cosmos-2.5` branch and signed `v0.2.0` tag. The former top-level `predict/`
and `transfer/` source trees are not part of the Cosmos 3 branch.

## Runtime Flow

```text
cosmos-h-surgical infer
        |
        v
cosmos_h_surgical.cli
        |
        v
cosmos_h_surgical.inference
        |
        +-- normalize supported surgical input compatibility
        +-- prepare a temporary checkpoint config view when needed
        +-- configure distributed timeout
        |
        v
pinned cosmos_framework.scripts.inference
```

The installed framework files are never overwritten. Compatibility behavior is
process-local and scoped to the inference command.

## Package Responsibilities

### `cli.py`

Provides the stable `cosmos-h-surgical` command and routes inference, training,
framework-provenance, and release-validation subcommands.

### `inference.py`

Delegates to the public framework CLI. It also supports the validated surgical
`resize_mode` contract, earlier release-candidate checkpoint configuration, and
the distributed timeout needed by the release test configuration.

### `checkpoints.py`

Loads model metadata from `release-manifest.json`. It will expose the public
model key after the v0.3.0 artifact is approved.

### `training.py` and `configs/`

Register project-owned surgical experiment configuration before invoking the
framework trainer. This interface is a developer preview for v0.3.0.

### `provenance.py` and `UPSTREAM.md`

Record the immutable public framework repository, revision, and approval
status. Framework changes require lockfile regeneration and compatibility
testing.

### `release.py`

Validates release metadata and rejects known internal path and URL markers.

## Ownership Boundary

This repository owns:

- Surgical model metadata and aliases
- Surgical input examples and documentation
- Surgical training configuration
- Narrow compatibility adapters required by approved checkpoints
- Release-manifest validation

The pinned Cosmos Framework owns model implementation, distributed inference,
generic input parsing, transfer operators, training infrastructure, and prompt
upsampling.

See [UPSTREAM.md](../UPSTREAM.md) for the revision and update policy, and the
[pinned upstream code-structure guide](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/code_structure.md)
for framework internals.
