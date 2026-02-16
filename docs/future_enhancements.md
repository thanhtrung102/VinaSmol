# Future Enhancement Issues

The following items from the technical report (Section 6) require training runs or further research, not code changes. They should be tracked as GitHub Issues.

## 1. Context Length Extension (2048 → 8192)

Training at 2048 sequence length caused the model to lose SmolLM2's 8192 context extension. Re-run context length extension following [Gao et al.](https://arxiv.org/abs/2410.02660), keeping the same data mixture but upsampling long documents.

**Labels:** enhancement, training

## 2. Ablation Study Without Vocabulary Extension

An ablation study without vocabulary extension would prove the benefits of vocab extension when the base model has zero target language capabilities. Compare:
- VinaSmol with vocabulary extension (current approach)
- VinaSmol without vocabulary extension (same training data, original tokenizer)

**Labels:** research, evaluation

## 3. Document-Level Code-Switching

Using a Vietnamese-to-English dictionary, inject code-switched segments into training documents to help the model align word representations between English and Vietnamese (as in [Sailor](https://arxiv.org/abs/2404.03608)).

**Labels:** enhancement, data

## 4. Temporal Dataset Alignment

Report the temporal distribution of the training dataset. The Binhvq news corpus, CulturaX, and MADLAD-400 date from 2023 or older, while FineWeb2-HQ was compiled in 2025. This age mismatch may degrade performance.

**Labels:** research, data

## 5. Extension to Lucie 7B

Apply the VinaSmol vocabulary extension methodology to [Lucie 7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1), a larger open-source model. This would require ReLoRA for efficient training.

**Labels:** enhancement, scaling

## 6. ReLoRA Integration

Integrate [ReLoRA](https://arxiv.org/abs/2307.05695) for EEVE stages 6 and 7 to enable training on more tokens with limited resources. Critical for scaling to Lucie 7B.

**Labels:** enhancement, training
