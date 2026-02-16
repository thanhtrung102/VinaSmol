# Continued Pre-Training

## Run

### First stage

#### Requirements

- A SmolLM2-360M checkpoint with extended vocabulary ([instructions here](../tokenization/README.md#extend-smollms-vocabulary-with-vietnamese))
- The prepared Vietnamese, English and code datasets ([instructions here](./dataset/README.md#prepare-data-for-training))


First, convert the SmolLM2-360M weights & tokenizer to a litgpt checkpoint. Read [this paragraph](../../docs/litgpt_help.md#convert-a-huggingface-transformers-checkpoint-to-a-litgpt-checkpoint) for more info.

```bash
litgpt convert_to_litgpt --model_name SmolLM2-360M vinasmol/tokenization/checkpoints/SmolLM2-360M_extended
```

Edit the files `vinasmol/tokenization/checkpoints/SmolLM2-360M_extended/model_config.yaml` and `cpt_stage_1_main.yml` to update the following fields:

```yml
model_config:
  name: VinaSmol-360M
  hf_config: {}
  ...
  vocab_size: 55936 # Adjust with the actual vocabulary size of the merged tokenizer
  padded_vocab_size: 55936
  ...

data:
  class_path: vinasmol_datamodule.VinaSmolData
  init_args:
    num_workers: 4 # Increase if GPU utilization is not 100%

train:
  global_batch_size: 128
  ...
  micro_batch_size: 2 # Should work with 20 GB VRAM, adjust if needed
  ...
  max_seq_length: 2048
```

The first stage of continued pretraining can then be started with the following commands:

```bash
cd vinasmol/training
# Start initial continued pretraining
# Consider running in the background, e.g. with nohup
bash cpt_stage_1_main.sh
```

#### Checkpoint conversion

Afterwards, convert the pretraining checkpoint for finetuning:

```bash
cd checkpoints/VinaSmol
# You can replace `final` by the last step of your run, e.g. step-000xxxxx
litgpt convert_pretrained_checkpoint cpt/VinaSmol_stage_1/final VinaSmol_stage_1
```

You can also [convert the model to HuggingFace](../../docs/litgpt_help.md#convert-a-litgpt-checkpoint-to-a-huggingface-transformers-checkpoint).

### Annealing stage

#### Requirements

- The [annealing data mixture](./dataset/README.md#annealing-datasets), prepared
- A model checkpoint from the initial pretraining phase

The annealing stage can be started with the following command:

```bash
cd vinasmol/training
bash cpt_stage_3_annealing.sh
```

## Training data

Details [here](./dataset/README.md).

## Framework

We use [litgpt](https://github.com/Lightning-AI/litgpt) for continued pretraining, which supports SmolLM2-360M.

In order to follow the multi-stage training of EEVE, we customize the [continued pretraining recipe of litgpt](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md#continued-pretraining-on-custom-data) by freezing the adequate parameters before the training starts.

## Recipe

The EEVE multi-stage training is implemented in [`eeve.py`](./eeve.py) with per-stage YAML configs and an orchestration script.

To run all 4 EEVE stages sequentially:

```bash
cd vinasmol/training
bash cpt_eeve_stages.sh
```

To resume from a specific stage (e.g., stage 6):

```bash
bash cpt_eeve_stages.sh 6
```

Each stage has its own config file (`cpt_eeve_stage_{3,4,6,7}.yml`) and initializes from the previous stage's final checkpoint.

### Vocabulary extension

We use the [Efficient and Effective Vocabulary Extension](https://arxiv.org/abs/2402.14714v1) method (EEVE), which encompasses both tokenization and training.

#### Tokenizer training

We train a new tokenizer on Vietnamese corpora and extend the base tokenizer with the new tokens. Details for running tokenizer training, merging, vocabulary extension and embedding initialization can be found [here](../tokenization/README.md).

#### Embedding initialization

For the input embeddings of the new Vietnamese tokens, we use the average embeddings of their subword tokens (default in the `tokenizers` library, as in [Hewitt](https://nlp.stanford.edu/~johnhew/vocab-expansion.html)). Whenever possible, we use a convex combination of the initialized embedding and the embedding of their translation using [EnViT5-base](https://huggingface.co/VietAI/envit5-translation).

Alternative: [transplant the embeddings of another model via Orthogonal Matching Pursuit](https://github.com/arcee-ai/mergekit/blob/main/docs/tokensurgeon.md)

Since SmolLM2-360M has tied embeddings due to its size, we simply propagate the input embeddings initialization to the output embeddings. This differs from the output embeddings initialization in EEVE.

<details>
<summary>Read more...</summary>
For the output embeddings of the new tokens, Kim et al. suggest to initialize them with the embeddings of the first subword token. This harmonization approach works if the base model has not its embeddings tied and has already some Vietnamese completion capabilities. Furthermore, it would be harmful for an English/Vietnamese model since most of the Vietnamese tokens start with a Latin consonant, which is already in the English alphabet. Single-consonant token embeddings already have a value in SmolLM2 and most of their information is useful for English completion, not Vietnamese completion.
</details>

### Multi-stage training

The different training stages used in EEVE are depicted below.

![EEVE pipeline](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0/resolve/main/EEVE_figure.png)

Refer to the [original paper](https://arxiv.org/abs/2402.14714v1) for more information.

Since SmolLM2-360M has tied embeddings, we only perform stages 3, 4, 6, 7 from the original EEVE pipeline.

### ReLoRA

In order to pretrain the model on even more tokens with limited resources, [ReLoRA](https://arxiv.org/abs/2307.05695) is an interesting technique that drastically cuts down on the compute and memory requirements.

We can use ReLoRA during stages 6 and 7 of the multi-stage training, where the number of trainable parameters increases with the transformer layers.

While ReLoRA may be unnecessary for very small LLMs such as [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), such an approach would be crucial for training larger models such as [Lucie-7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1).

Implementations (not reviewed yet):
- https://github.com/ElleLeonne/Lightning-ReLoRA
- https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/llama-2/relora.yml

How to integrate both EEVE and ReLoRA into existing training frameworks remains very unclear. Custom training code is very likely to be necessary, therefore a PyTorch Lightning-based solution could be more suitable for customization.

### Hyperparameters

| | SmolLM2-360M | VinaSmol-360M |
| ---------------- | ----------- | ----------- |
| **Total tokens** | 3T | 2B (continued) |
| **Sequence length** | 2048* | 2048 |
| **Global Batch size** | 1024 (2M tokens) | 32-256 |
| **Learning rate** | 3e-3 | 1e-4 / 2e-4 |
| **Warmup steps** | 5000 | 50 |

\* 8192 after context length extension.

Refer to https://huggingface.co/blog/smollm and the [SmolLM2 paper](https://arxiv.org/abs/2502.02737v1) for hyperparameter tuning.

Similar to [Sailor 7B](https://arxiv.org/abs/2404.03608), we adjust the language mixture proportions and the learning rate based on initial experiments.

Further improvements can be done to adjust batch size and learning rate.

### Context extension

We continued the pretraining of SmolLM2 on sequence lengths of 2048 to save costs. However, this caused the model to lose its context length extension of 8192. Therefore we rerun context length extension for the final VinaSmol model.

We follow the procedure outlined by [Gao et al.](https://arxiv.org/abs/2410.02660), keeping the same data mixture but upsampling long documents within each dataset.

### Later stages and annealing

Following the approach used by [SmolLM2](https://arxiv.org/abs/2502.02737v1), we add high-quality content, technical content, textbooks, medium-sized and [instruction](https://magazine.sebastianraschka.com/p/instruction-pretraining-llms#%C2%A7pretraining-with-instruction-data) datasets during the annealing phase in order to maximize their impact.

## Further improvements

### Larger models

Using a larger base model such as [Lucie 7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1) may improve the model capabilities and degree of multilinguality.

Techniques such as [depth upscaling](https://planetbanatt.net/articles/modelmerging.html#orgf613f37) used by [SOLAR 10.7B](https://arxiv.org/abs/2312.15166) could be used to increase the size of the model and scale to a higher number of parameters. However it's unclear whether depth upscaling coud prove to be useful for very small models.

## Citations

- [litgpt, 2023](https://github.com/Lightning-AI/litgpt)
