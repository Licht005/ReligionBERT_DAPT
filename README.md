# ReligionBERT

Domain-adaptive pre-training of BERT on Bible corpora, with fine-tuning, cross-lingual and perplexity evaluation.

## Setup

```
pip install -r requirements.txt
export RELIGIONBERT_HOME=/path/to/ReligionBERT   # all data, models and results live here
```

On Colab, mount Drive first and point `RELIGIONBERT_HOME` at `/content/drive/MyDrive/ReligionBERT`.
Set `REPORT_TO=wandb` to log runs to Weights & Biases (off by default).

## Scripts

| File | Purpose |
|---|---|
| `common.py` | Paths, constants, book code map, JSON helpers |
| `trainer_utils.py` | Checkpoint cleanup callback and shared `TrainingArguments` |
| `prepare_corpus.py` | Clone corpus, extract verses, tokenise for pre-training |
| `pretrain.py` | Continued MLM pre-training (ReligionBERT, MultiReligionBERT) |
| `build_datasets.py` | Semantic similarity and book classification datasets |
| `generate_qa.py` | Extractive QA dataset via Groq (needs `GROQ_API_KEY`) |
| `finetune.py` | Fine-tune and test one variant on one task |
| `perplexity.py` | Held-out masked-LM perplexity |
| `crosslingual.py` | Zero-shot book classification on five African languages |
| `plot_results.py` | All figures from saved metrics |

## Run order

```
python prepare_corpus.py
python pretrain.py --corpus english
python pretrain.py --corpus multilingual
python build_datasets.py
python generate_qa.py

for v in bert religion-bert mbert multi-religion-bert; do
  for t in sim cls qa; do python finetune.py --task $t --variant $v; done
done
python finetune.py --task cls --variant xlmr

python perplexity.py
python crosslingual.py
python plot_results.py
```

To skip pre-training, pass the published weights, for example
`python finetune.py --task cls --variant religion-bert --model-path LucasLicht/religion-bert`.
Metrics are written to `results/metrics`, figures to `results/figures`.
