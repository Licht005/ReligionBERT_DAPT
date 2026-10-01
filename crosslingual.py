"""Zero-shot cross-lingual book classification on African-language Bibles.

Needs English classifiers first:
    python finetune.py --task cls --variant mbert
    python finetune.py --task cls --variant multi-religion-bert
    python finetune.py --task cls --variant xlmr
"""
import random
from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (BOOK_NAMES, FINETUNE_DATA, METRICS, MODELS, SEED, load_verse_map, read_json, write_json)

TARGET_LANGUAGES = {'Amharic': 'Amharic', 'Shona': 'Shona', 'Xhosa': 'Xhosa', 'Ewe': 'Ewe-NT', 'Swahili': 'Swahili-NT'}
EVAL_VARIANTS = ['mbert', 'multi-religion-bert', 'xlmr']
SAMPLES_PER_LANGUAGE = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_eval_set(corpus, label_map):
    """Roughly 300 verses per language, spread evenly over the books the corpus contains."""
    by_book = defaultdict(list)
    for vid, text in load_verse_map(corpus).items():
        parts = vid.split('.')
        book = BOOK_NAMES.get(parts[1]) if len(parts) >= 3 else None
        if book in label_map:
            by_book[book].append({'text': text.strip(), 'label': label_map[book]})
    per_book = max(1, SAMPLES_PER_LANGUAGE // len(by_book))
    samples = [s for verses in by_book.values() for s in random.sample(verses, min(per_book, len(verses)))]
    random.shuffle(samples)
    return samples


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size=64):
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors='pt', truncation=True,
                        max_length=128, padding='max_length').to(device)
        preds += model(**enc).logits.argmax(dim=-1).tolist()
    return preds


def main():
    random.seed(SEED)
    label_map = read_json(FINETUNE_DATA / 'classification' / 'label_map.json')
    eval_sets = {lang: build_eval_set(corpus, label_map) for lang, corpus in TARGET_LANGUAGES.items()}
    for lang, samples in eval_sets.items():
        print(f'{lang}: {len(samples)} samples')

    results = {}
    for variant in EVAL_VARIANTS:
        path = MODELS / 'finetuned' / 'cls' / variant
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        results[variant] = {}
        for lang, samples in eval_sets.items():
            labels = [s['label'] for s in samples]
            preds = predict(model, tokenizer, [s['text'] for s in samples])
            results[variant][lang] = {'accuracy': accuracy_score(labels, preds),
                                      'macro_f1': f1_score(labels, preds, average='macro')}
            print(f'{variant} on {lang}: {results[variant][lang]}')
        del model
        torch.cuda.empty_cache()

    write_json(METRICS / 'crosslingual.json', results, indent=2)


if __name__ == '__main__':
    main()
