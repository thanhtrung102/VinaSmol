# Contributing to VinaSmol

We welcome contributions to VinaSmol! This document provides guidelines for contributing to the project.

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and initialize submodules:

```bash
git clone https://github.com/slivering/VinaSmol-Unaite.git
cd VinaSmol-Unaite
git submodule update --init --recursive
```

3. Set up the virtual environment:

```bash
uv lock
uv sync --all-packages
```

4. Copy and configure environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

- `vinasmol/` — Main package (tokenization, training, finetuning, evaluation)
- `ccvj/` — CreativeCommons Vietnamese Journals dataset tool
- `vinasmol_datamodule/` — Custom LitGPT data loader
- `scripts/` — Conversion utilities
- `tests/` — Test suite

## How to Contribute

1. **Open an issue** to discuss your proposed change before starting work.
2. **Fork the repository** and create a new branch from `main`.
3. **Make your changes** following the code style guidelines.
4. **Add tests** for any new functionality.
5. **Run the test suite** to ensure nothing is broken.
6. **Submit a pull request** with a clear description of your changes.

## Areas for Contribution

- Dataset curation and filtering improvements
- Training methodology enhancements
- Evaluation benchmarks and scripts
- Documentation improvements
- Bug fixes

## Questions?

Feel free to open an issue to get started.
