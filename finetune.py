"""Fine-tune one model variant on one downstream task and record test metrics.

    python finetune.py --task sim --variant religion-bert
    python finetune.py --task cls --variant mbert
    python finetune.py --task qa  --variant bert

Tasks: sim (semantic similarity), cls (66-class book classification), qa (extractive QA).
Variants: bert, religion-bert, mbert, multi-religion-bert (and xlmr for cls).
"""
import argparse

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForQuestionAnswering, AutoModelForSequenceClassification,
                          AutoTokenizer, DefaultDataCollator, Trainer)

from common import CHECKPOINTS, FINETUNE_DATA, METRICS, MODELS, VARIANTS, read_json, write_json
from trainer_utils import PermanentDeleteCallback, make_args

MAX_LENGTH = 128
QA_MAX_LENGTH = 384
QA_STRIDE = 128


def epoch_args(task, variant, name, **kwargs):
    return make_args(
        CHECKPOINTS / 'finetuning' / f'{task}_{variant}{name}', f'{variant}-{task}{name}',
        per_device_train_batch_size=32, per_device_eval_batch_size=32,
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        greater_is_better=True, **kwargs,
    )


def run_trainer(model, tokenizer, args, train, val, metrics_fn):
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      compute_metrics=metrics_fn, processing_class=tokenizer,
                      callbacks=[PermanentDeleteCallback()])
    trainer.train()
    return trainer


# Semantic similarity

def encode_pairs(path, tokenizer):
    data = read_json(path)
    enc = tokenizer([d['sentence1'] for d in data], [d['sentence2'] for d in data],
                    truncation=True, max_length=MAX_LENGTH, padding='max_length')
    enc['labels'] = [float(d['score']) for d in data]
    return Dataset.from_dict(enc)


def sim_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.flatten()
    return {'pearson': float(pearsonr(preds, labels)[0]), 'spearman': float(spearmanr(preds, labels)[0])}


def finetune_sim(variant, model_path):
    """Stage 1: balanced subset, 5 epochs at 2e-5. Stage 2: full set, 2 epochs at 1e-5."""
    data_dir = FINETUNE_DATA / 'semantic_similarity'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    val, test = (encode_pairs(data_dir / f'{s}.json', tokenizer) for s in ['val', 'test'])

    for stage, (split, lr, epochs) in enumerate([('train_balanced', 2e-5, 5), ('train', 1e-5, 2)], start=1):
        args = epoch_args('sim', variant, f'_stage{stage}', num_train_epochs=epochs, learning_rate=lr,
                          warmup_ratio=0.1, metric_for_best_model='pearson')
        trainer = run_trainer(model, tokenizer, args, encode_pairs(data_dir / f'{split}.json', tokenizer),
                              val, sim_metrics)

    result = trainer.evaluate(test)
    return model, tokenizer, {'pearson': result['eval_pearson'], 'spearman': result['eval_spearman']}


# Book classification

def encode_books(path, tokenizer, label_map):
    data = read_json(path)
    enc = tokenizer([d['text'] for d in data], truncation=True, max_length=MAX_LENGTH, padding='max_length')
    enc['labels'] = [label_map[d['book_name']] for d in data]
    return Dataset.from_dict(enc)


def cls_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'accuracy': accuracy_score(labels, preds), 'macro_f1': f1_score(labels, preds, average='macro')}


def finetune_cls(variant, model_path):
    data_dir = FINETUNE_DATA / 'classification'
    label_map = read_json(data_dir / 'label_map.json')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(label_map), id2label={i: n for n, i in label_map.items()},
        label2id=label_map, ignore_mismatched_sizes=True)
    train, val, test = (encode_books(data_dir / f'{s}.json', tokenizer, label_map)
                        for s in ['train', 'val', 'test'])

    warmup = {'warmup_steps': 100} if variant == 'xlmr' else {'warmup_ratio': 0.1}
    args = epoch_args('cls', variant, '', num_train_epochs=10, learning_rate=2e-5,
                      metric_for_best_model='macro_f1', **warmup)
    trainer = run_trainer(model, tokenizer, args, train, val, cls_metrics)

    result = trainer.evaluate(test)
    return model, tokenizer, {'accuracy': result['eval_accuracy'], 'macro_f1': result['eval_macro_f1']}


# Extractive QA

def encode_qa(data, tokenizer):
    """Tokenise question/context pairs and map each answer's character span to token positions."""
    enc = tokenizer([d['question'] for d in data], [d['context'] for d in data],
                    truncation='only_second', max_length=QA_MAX_LENGTH, stride=QA_STRIDE,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')
    sample_map = enc.pop('overflow_to_sample_mapping')
    offset_map = enc.pop('offset_mapping')
    starts, ends = [], []

    for i, offsets in enumerate(offset_map):
        example = data[sample_map[i]]
        cls_index = enc['input_ids'][i].index(tokenizer.cls_token_id)
        seq_ids = enc.sequence_ids(i)
        ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)
        ctx_end = next(j for j, s in reversed(list(enumerate(seq_ids))) if s == 1)
        char_start = example['answer_start']
        char_end = char_start + len(example['answer'])

        if offsets[ctx_start][0] > char_end or offsets[ctx_end][1] < char_start:
            starts.append(cls_index)
            ends.append(cls_index)
            continue

        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= char_start:
            tok_start += 1
        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= char_end:
            tok_end -= 1
        starts.append(tok_start - 1)
        ends.append(tok_end + 1)

    enc['start_positions'] = starts
    enc['end_positions'] = ends
    return Dataset.from_dict(dict(enc))


def token_f1(pred, gold):
    p, g = set(pred.split()), set(gold.split())
    if not p or not g:
        return float(p == g)
    common = len(p & g)
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def evaluate_qa(model, tokenizer, data):
    """Exact match and token F1 (percent) from the argmax start and end positions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    exact, f1 = [], []
    for item in data:
        inputs = tokenizer(item['question'], item['context'], return_tensors='pt', truncation=True,
                           max_length=QA_MAX_LENGTH, padding='max_length').to(device)
        with torch.no_grad():
            out = model(**inputs)
        start, end = out.start_logits.argmax().item(), out.end_logits.argmax().item()
        pred = ''
        if end >= start:
            pred = tokenizer.decode(inputs['input_ids'][0][start:end + 1], skip_special_tokens=True)
        pred, gold = pred.strip().lower(), item['answer'].strip().lower()
        exact.append(int(pred == gold))
        f1.append(token_f1(pred, gold))
    return {'exact_match': float(np.mean(exact) * 100), 'f1': float(np.mean(f1) * 100)}


def finetune_qa(variant, model_path):
    data_dir = FINETUNE_DATA / 'qa'
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # must be a fast tokenizer for offsets
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    train = encode_qa(read_json(data_dir / 'train.json'), tokenizer)

    args = make_args(CHECKPOINTS / 'finetuning' / f'qa_{variant}', f'{variant}-qa',
                     num_train_epochs=4, per_device_train_batch_size=8, learning_rate=1e-5,
                     warmup_steps=50, logging_steps=10, save_strategy='no')
    Trainer(model=model, args=args, train_dataset=train, data_collator=DefaultDataCollator(),
            processing_class=tokenizer).train()

    return model, tokenizer, evaluate_qa(model, tokenizer, read_json(data_dir / 'test.json'))


TASKS = {'sim': finetune_sim, 'cls': finetune_cls, 'qa': finetune_qa}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--model-path', help='override the model location, e.g. a HuggingFace Hub id')
    args = parser.parse_args()

    model, tokenizer, metrics = TASKS[args.task](args.variant, args.model_path or VARIANTS[args.variant])
    print(f'{args.variant} {args.task} test: {metrics}')

    save_dir = MODELS / 'finetuned' / args.task / args.variant
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    write_json(METRICS / f'{args.variant}_{args.task}.json', metrics, indent=2)


if __name__ == '__main__':
    main()
