"""Continued masked-language-model pre-training on the Bible corpus.

    python pretrain.py --corpus english         # ReligionBERT
    python pretrain.py --corpus multilingual    # MultiReligionBERT

Rerun the same command to resume from the latest checkpoint after a session reset.
"""
import argparse

from datasets import concatenate_datasets, load_from_disk
from transformers import (BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling, Trainer)

from common import CHECKPOINTS, HUB_IDS, METRICS, MODELS, PROCESSED, SEED, VARIANTS, write_json
from trainer_utils import PermanentDeleteCallback, make_args

CORPORA = {
    'english': {'base': VARIANTS['bert'], 'test_size': 0.05, 'output': 'religion-bert'},
    'multilingual': {'base': VARIANTS['mbert'], 'test_size': 0.02, 'output': 'multi-religion-bert'},
}


def load_splits(corpus, test_size):
    chunks = sorted((PROCESSED / f'tokenized_{corpus}').glob('chunk_*'),
                    key=lambda p: int(p.name.split('_')[1]))
    dataset = concatenate_datasets([load_from_disk(str(c)) for c in chunks])
    return dataset.remove_columns(['token_type_ids']).train_test_split(test_size=test_size, seed=SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=CORPORA, required=True)
    parser.add_argument('--push', action='store_true', help='push the final model to the HuggingFace Hub')
    args = parser.parse_args()
    cfg = CORPORA[args.corpus]

    tokenizer = BertTokenizer.from_pretrained(cfg['base'])
    model = BertForMaskedLM.from_pretrained(cfg['base'])
    splits = load_splits(args.corpus, cfg['test_size'])

    output_dir = CHECKPOINTS / f'pretraining-{args.corpus}'
    training_args = make_args(
        output_dir, f'religion-bert-{args.corpus}',
        max_steps=30_000, per_device_train_batch_size=16, per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, learning_rate=3e-5, warmup_steps=500, logging_steps=100,
        eval_strategy='steps', eval_steps=500, save_strategy='steps', save_steps=500,
        load_best_model_at_end=True, metric_for_best_model='eval_loss',
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=splits['train'], eval_dataset=splits['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
        processing_class=tokenizer, callbacks=[PermanentDeleteCallback()],
    )
    has_checkpoint = any(output_dir.glob('checkpoint-*'))
    trainer.train(resume_from_checkpoint=True if has_checkpoint else None)

    save_dir = MODELS / cfg['output']
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    write_json(METRICS / f'pretrain_{args.corpus}.json', trainer.state.log_history)
    if args.push:
        model.push_to_hub(HUB_IDS[args.corpus])
        tokenizer.push_to_hub(HUB_IDS[args.corpus])
    print(f'Saved {save_dir}')


if __name__ == '__main__':
    main()
