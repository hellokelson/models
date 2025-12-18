# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Merlin Models is NVIDIA's library for recommender system models, providing high-quality implementations of both classic and state-of-the-art deep learning architectures. The library supports both TensorFlow (primary) and PyTorch (partial), with a focus on GPU-accelerated training and inference.

Key characteristics:
- Schema-driven model architecture that adapts to input features automatically
- Block-based composable design for building custom models
- Tight integration with the Merlin ecosystem (NVTabular, Core, DataLoader, Systems)
- Support for both retrieval models (Matrix Factorization, Two-Tower, YouTube DNN) and ranking models (DLRM, DCN-v2, DeepFM)

## Development Commands

### Installation
```bash
# Install with all dependencies (recommended for development)
pip install -e .[all]

# Install specific framework extras
pip install -e .[tensorflow-dev]  # TensorFlow development
pip install -e .[pytorch-dev]     # PyTorch development
```

### Testing

The test suite uses pytest markers to target specific frameworks and test types:

```bash
# Run all tests
make tests

# Run TensorFlow tests (most common during development)
make tests-tf                    # Unit tests only, parallel execution
make tests-tf-examples          # Example/notebook tests
make tests-tf-integration       # Integration tests

# Run PyTorch tests
make tests-torch

# Run tests for specific ML backends
make tests-implicit             # Implicit library tests
make tests-lightfm              # LightFM tests
make tests-xgboost              # XGBoost tests

# Run tests for changed files only (faster iteration)
make tests-tf-changed

# Run single test file or test
pytest tests/unit/tf/models/test_ranking.py
pytest tests/unit/tf/models/test_ranking.py::test_dlrm_model -v

# Run tests with specific markers
pytest -m "tensorflow and unit" tests/
pytest -m "torch" tests/
```

### Linting and Code Quality

```bash
# Run all linters
make lint

# Individual linters
flake8 .
black --check .
isort -c .

# Auto-fix formatting
black .
isort .
```

### Documentation

```bash
# Build documentation
make docs                        # Builds and serves at localhost:8000
cd docs && make html            # Build only

# Update API docs
make docstrings
```

## Architecture

### Dual Framework Structure

The codebase maintains parallel implementations for TensorFlow and PyTorch:

- **TensorFlow**: `merlin/models/tf/` - Primary, fully-featured implementation
- **PyTorch**: `merlin/models/torch/` - Partial implementation, focus on transformer-based models

Each framework directory contains:
- `blocks/` - Reusable model components (MLP, DLRM, Cross, Experts, Retrieval, Sampling)
- `inputs/` - Input layer implementations (embeddings, continuous features)
- `outputs/` - Output layer and prediction task implementations
- `models/` - Complete model implementations (base, ranking, retrieval)
- `core/` - Core abstractions (Block, combinators, encoders)
- `metrics/`, `losses/`, `transforms/` - Supporting components

### Schema-Driven Design

Models in Merlin are built around the concept of a `Schema` object (from `merlin.schema`):

1. **Schema Creation**: NVTabular creates schemas during feature engineering that identify continuous features, categorical features, and targets
2. **Automatic Input Layers**: Models use the schema to automatically create appropriate input layers and embedding tables
3. **Feature Inference**: Prediction tasks can infer target features from schema tags

Example workflow:
```python
import merlin.models.tf as mm
from merlin.io.dataset import Dataset

train = Dataset(PATH)  # Dataset includes schema
model = mm.DLRMModel(
    train.schema,  # Schema determines inputs automatically
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(train.schema)
)
```

### Block-Based Composition

Models are composed of reusable "blocks":

- **Input Blocks**: `InputBlock`, `InputBlockV2` - Handle feature inputs and embeddings
- **Core Blocks**: `MLPBlock`, `CrossBlock`, `DLRMBlock` - Neural network architectures
- **Retrieval Blocks**: `TwoTowerBlock`, `MatrixFactorizationBlock`, `DualEncoderBlock`
- **Expert Blocks**: `MMOEBlock`, `PLEBlock`, `CGCBlock` - Multi-task learning
- **Combinators**: `SequentialBlock`, `ParallelBlock`, `ResidualBlock` - Block composition

Blocks can be combined to create custom architectures while maintaining compatibility with the schema-driven approach.

### Key Model Types

**Ranking Models** (in `merlin/models/tf/models/ranking.py`):
- `MatrixFactorizationModel` - Classic collaborative filtering
- `DLRMModel` - Deep Learning Recommendation Model
- `DCNModel` - Deep & Cross Network
- `DeepFMModel` - Factorization Machine with deep component

**Retrieval Models** (in `merlin/models/tf/models/retrieval.py`):
- `TwoTowerModel` - Dual encoder for candidate retrieval
- `YouTubeDNNModel` - YouTube-style retrieval model
- `MatrixFactorizationModel` - Can be used for retrieval

### Integration Points

**NVTabular Integration**: Models expect datasets preprocessed with NVTabular, which provides:
- Schema with feature metadata
- Parquet-based datasets optimized for GPU loading
- Feature engineering transformations

**DataLoader Integration**: `merlin.models.tf.loader.Loader` provides optimized GPU data loading:
- Processes large chunks instead of individual samples
- Streams from disk for datasets larger than memory
- Asynchronous batch preparation

**Merlin Systems Integration**: Trained models can be exported for serving via Merlin Systems (Triton Inference Server integration).

## Testing Strategy

Tests are organized by type and framework:

- `tests/unit/tf/` - TensorFlow unit tests (blocks, models, inputs, outputs)
- `tests/unit/torch/` - PyTorch unit tests
- `tests/integration/` - Cross-component integration tests
- `tests/common/` - Shared test utilities

**Pytest Markers** (defined in `pytest.ini`):
- Framework: `tensorflow`, `torch`, `implicit`, `lightfm`, `xgboost`
- Type: `unit`, `integration`, `notebook`, `examples`
- Special: `horovod` (distributed training), `multigpu`, `singlegpu`
- CI: `changed` (only changed files), `always` (always run)

## Common Development Workflows

### Adding a New Block (TensorFlow)

1. Create block class in appropriate `merlin/models/tf/blocks/` file
2. Inherit from `Block` or `TabularBlock` base class
3. Implement `call()` method with signature: `call(inputs, **kwargs)`
4. Export in `merlin/models/tf/__init__.py`
5. Add tests in `tests/unit/tf/blocks/`
6. Update `docs/source/api.rst` with reference to new class

### Adding a New Model

1. Implement in `merlin/models/tf/models/` (typically `ranking.py` or `retrieval.py`)
2. Inherit from `BaseModel` or `RetrievalModel`
3. Models should accept `schema` as first argument
4. Export from `merlin/models/tf/__init__.py`
5. Add example usage in `examples/` directory
6. Update API documentation in `docs/source/api.rst`

### Running Tests in CI

GitHub Actions workflows in `.github/workflows/`:
- `gpu.yml` - GPU unit tests
- `gpu-multi.yml` - Multi-GPU and Horovod tests
- `cpu-*.yml` - Cross-component compatibility tests with other Merlin libraries
- `lint.yml` - Code quality checks

## Important Notes

### TensorFlow Memory Configuration

The library automatically configures TensorFlow memory allocation via `configure_tensorflow()` (called in `merlin/models/tf/__init__.py`). This prevents TensorFlow from allocating all GPU memory upfront.

### Multi-GPU and Distributed Training

- Horovod support is available but requires special installation (see `requirements/horovod.txt`)
- Multi-GPU tests require `multigpu` marker and special test environments
- Use `make tests-tf` for single-GPU development; multi-GPU tests run in CI

### PyTorch Status

PyTorch support is partial and focused on specific use cases. For comprehensive transformer-based session models, see the separate Transformers4Rec library. The PyTorch API here provides core blocks and some models but is not feature-complete with the TensorFlow API.

### Documentation Warnings

When adding/modifying classes, always update `docs/source/api.rst`. The docs build will emit warnings for:
- Missing classes referenced in `api.rst`
- Malformed docstrings
- Broken cross-references

Run `make docs` locally to catch these before pushing.
