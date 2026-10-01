"""Masked-LM perplexity of generic BERT and ReligionBERT on English Bible verses.

    python perplexity.py
"""
import argparse
import math

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from common import METRICS, SEED, VARIANTS, load_verse_map, write_json


@torch.no_grad()
def perplexity(model_path, texts, device, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()
    special = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    torch.manual_seed(SEED)  # identical masked positions for every model

    total_loss, total_tokens = 0.0, 0
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length,
                           padding='max_length').to(device)
        ids = inputs['input_ids']
        masked = (torch.rand(ids.shape, device=device) < 0.15) & ~torch.isin(ids, torch.tensor(special, device=device))
        n = masked.sum().item()
        if n == 0:
            continue
        labels = ids.masked_fill(~masked, -100)
        inputs['input_ids'] = ids.masked_fill(masked, tokenizer.mask_token_id)
        total_loss += model(**inputs, labels=labels).loss.item() * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-verses', type=int, default=500)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    texts = list(load_verse_map('English').values())[-args.num_verses:]
    results = {}
    for variant in ['bert', 'religion-bert']:
        results[variant] = round(perplexity(VARIANTS[variant], texts, device), 3)
        print(f'{variant}: {results[variant]}')
    write_json(METRICS / 'perplexity.json', results, indent=2)


if __name__ == '__main__':
    main()
