"""Shared paths, constants, and small helpers used by every script."""
import json
import os
from pathlib import Path

SEED = 42
REPORT_TO = os.environ.get('REPORT_TO', 'none')  # set REPORT_TO=wandb to log runs to Weights & Biases

BASE = Path(os.environ.get('RELIGIONBERT_HOME', 'ReligionBERT'))
RAW_BIBLE = BASE / 'datasets/raw/bible'
BIBLE_XML = RAW_BIBLE / 'bibles'
PROCESSED = BASE / 'datasets/processed'
FINETUNE_DATA = BASE / 'datasets/finetuning'
MODELS = BASE / 'models'
CHECKPOINTS = BASE / 'checkpoints'
METRICS = BASE / 'results/metrics'
FIGURES = BASE / 'results/figures'

ENGLISH_CORPUS = ['English', 'English-WEB']
MULTILINGUAL_CORPUS = ENGLISH_CORPUS + [
    'French', 'Spanish', 'Portuguese', 'German', 'Amharic',
    'Shona', 'Xhosa', 'Malagasy', 'Somali', 'Zarma',
]
EVAL_ONLY_CORPUS = ['Ewe-NT', 'Swahili-NT']

HUB_IDS = {
    'english': 'LucasLicht/religion-bert',
    'multilingual': 'LucasLicht/multi-religion-bert',
}

VARIANTS = {
    'bert': 'bert-base-uncased',
    'religion-bert': str(MODELS / 'religion-bert'),
    'mbert': 'bert-base-multilingual-cased',
    'multi-religion-bert': str(MODELS / 'multi-religion-bert'),
    'xlmr': 'xlm-roberta-base',
}
DISPLAY_NAMES = {
    'bert': 'Generic BERT',
    'religion-bert': 'ReligionBERT',
    'mbert': 'mBERT',
    'multi-religion-bert': 'MultiReligionBERT',
    'xlmr': 'XLM-R',
}

# Codes follow the christos-c corpus (MAR, JOH, PHI, 1JO ...), not the usual MRK, JHN, PHP, 1JN.
BOOK_NAMES = {
    'GEN': 'Genesis', 'EXO': 'Exodus', 'LEV': 'Leviticus', 'NUM': 'Numbers',
    'DEU': 'Deuteronomy', 'JOS': 'Joshua', 'JDG': 'Judges', 'RUT': 'Ruth',
    '1SA': '1 Samuel', '2SA': '2 Samuel', '1KI': '1 Kings', '2KI': '2 Kings',
    '1CH': '1 Chronicles', '2CH': '2 Chronicles', 'EZR': 'Ezra', 'NEH': 'Nehemiah',
    'EST': 'Esther', 'JOB': 'Job', 'PSA': 'Psalms', 'PRO': 'Proverbs',
    'ECC': 'Ecclesiastes', 'SON': 'Song of Solomon', 'ISA': 'Isaiah',
    'JER': 'Jeremiah', 'LAM': 'Lamentations', 'EZE': 'Ezekiel', 'DAN': 'Daniel',
    'HOS': 'Hosea', 'JOE': 'Joel', 'AMO': 'Amos', 'OBA': 'Obadiah', 'JON': 'Jonah',
    'MIC': 'Micah', 'NAH': 'Nahum', 'HAB': 'Habakkuk', 'ZEP': 'Zephaniah',
    'HAG': 'Haggai', 'ZEC': 'Zechariah', 'MAL': 'Malachi', 'MAT': 'Matthew',
    'MAR': 'Mark', 'LUK': 'Luke', 'JOH': 'John', 'ACT': 'Acts', 'ROM': 'Romans',
    '1CO': '1 Corinthians', '2CO': '2 Corinthians', 'GAL': 'Galatians',
    'EPH': 'Ephesians', 'PHI': 'Philippians', 'COL': 'Colossians',
    '1TH': '1 Thessalonians', '2TH': '2 Thessalonians', '1TI': '1 Timothy',
    '2TI': '2 Timothy', 'TIT': 'Titus', 'PHM': 'Philemon', 'HEB': 'Hebrews',
    'JAM': 'James', '1PE': '1 Peter', '2PE': '2 Peter', '1JO': '1 John',
    '2JO': '2 John', '3JO': '3 John', 'JUD': 'Jude', 'REV': 'Revelation',
}
OT_BOOKS = set(list(BOOK_NAMES)[:39])  # BOOK_NAMES is in canonical order


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_json(path, obj, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_verse_map(name):
    """Return {verse_id: text} for a processed Bible, in corpus order."""
    return {v['id']: v['text'] for v in read_json(PROCESSED / f'{name}.json')}


def book_code(verse_id):
    """Verse ids look like b.GEN.1.1, so the book code is the second field."""
    return verse_id.split('.')[1]


def split_80_10_10(items):
    n = len(items)
    return {'train': items[:int(n * 0.8)],
            'val': items[int(n * 0.8):int(n * 0.9)],
            'test': items[int(n * 0.9):]}
