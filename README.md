# VinaSmol: Extending SmolLM with Vietnamese

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

VinaSmol is a proof-of-concept project that aims to add Vietnamese language capabilities and knowledge to [SmolLM 360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).

Our approach aims to demonstrate that efficient pretraining methods on an open, medium-sized Vietnamese dataset (less than 10B tokens) can effectively integrate a new language into an LLM that was not trained on it.

Future plans include extending these techniques to [Lucie 7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1).

> [!NOTE]
> 
> For reviewers, a technical report for VinaSmol is available [here](./report.pdf). The report is in draft version as the project is still being worked on.

## Disclaimer

This model is still in heavy development and is not ready for general use.

## Development

> [!NOTE]
>
> This section is aimed at developers who want to replicate VinaSmol or contribute to the project.

Install [uv](https://docs.astral.sh/uv/) and set up the virtual environment:

```bash
uv lock
uv sync --all-packages
```

We use the [LitGPT](https://github.com/Lightning-AI/litgpt) framework for training and finetuning. Make sure to check their documentation and some specific [tips](./docs/litgpt_help.md) for VinaSmol compatibility.

VinaSmol can be fully replicated by following the steps in the order below.

## Training methods

We employ a comprehensive pipeline to integrate the Vietnamese language into SmolLM. We outline the main steps below:

1. [Extend an existing model’s tokenizer with new Vietnamese tokens](./vinasmol/tokenization/README.md).
2. [Continued pre-training of the model on Vietnamese](./vinasmol/training/README.md)
3. [Fine-tuning and merging for instructions, capabilities and safety](./vinasmol/finetuning/README.md)
4. [Evaluation of the final model in both Vietnamese and English](./vinasmol/evaluation/README.md)

We combine state-of-the-art techniques used for LLMs targeting South-East Asian languages, prioritizing efficiency and results within reasonable compute resources. The goal is to enable replication of the VinaSmol training process on a single node and in reasonable time.

## Related work

Existing state-of-the-art Vietnamese models are developed by organizations with larger resources, often targeting South-East Asian languages as a whole. Few models have been developed with frugal training in mind.

Furthermore, there is a lack of fully open-source and performant Vietnamese models. Indeed, most of the available models are based on Qwen, Llama or Gemma which do not disclose their training datasets. The authors of other base Vietnamese models often remain vague on the provenance of their training data.

## Future directions

We are looking forward further enhancements for VinaSmol:

- Better dataset curation
- Improved reasoning abilities
- Safety-aware pretraining

## Contribution

VinaSmol is still in development. We welcome any kind of contribution, including suggestions concerning our training methodology. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and feel free to open an issue to get started.

## Citation

```bibtex
@techreport{vutu2025vinasmol,
  title={VinaSmol: Efficient Extension of an English-only SLM to Vietnamese},
  author={Vu Tu, Linh},
  year={2025},
  institution={\'Ecole polytechnique, LINAGORA},
  note={Technical report (unreleased)}
}
```
