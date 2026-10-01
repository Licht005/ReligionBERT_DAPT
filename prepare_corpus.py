"""Download the Bible corpus, extract verses, and build the tokenised pre-training data.

    python prepare_corpus.py                 # everything
    python prepare_corpus.py --no-tokenize   # verse JSON and text files only
"""
import argparse
import subprocess
import xml.etree.ElementTree as ET

from common import (BIBLE_XML, ENGLISH_CORPUS, EVAL_ONLY_CORPUS, MULTILINGUAL_CORPUS,
                    PROCESSED, RAW_BIBLE, VARIANTS, read_json, write_json)

CORPUS_URL = 'https://github.com/christos-c/bible-corpus.git'
MAX_LENGTH = 128
CHUNK_SIZE = 50_000


def extract_verses(xml_path):
    body = ET.parse(xml_path).getroot()[1][0]
    verses = []
    for book in body:
        for chapter in book:
            for seg in chapter:
                text = (seg.text or '').strip()
                if seg.attrib.get('type') == 'verse' and text:
                    verses.append({'id': seg.attrib.get('id', ''), 'text': text})
    return verses


def write_text_corpus(names, out_path):
    with open(out_path, 'w', encoding='utf-8') as out:
        for name in names:
            for verse in read_json(PROCESSED / f'{name}.json'):
                out.write(verse['text'] + '\n')


def tokenize_corpus(txt_path, tokenizer, out_dir):
    """Tokenise in chunks so the multilingual corpus fits in Colab RAM."""
    from datasets import Dataset

    with open(txt_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f'{txt_path.name}: {len(lines):,} lines')
    for i in range(0, len(lines), CHUNK_SIZE):
        encoded = tokenizer(lines[i:i + CHUNK_SIZE], truncation=True, max_length=MAX_LENGTH,
                            padding='max_length', return_special_tokens_mask=True)
        Dataset.from_dict(encoded).save_to_disk(str(out_dir / f'chunk_{i // CHUNK_SIZE}'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-tokenize', action='store_true')
    args = parser.parse_args()

    if not BIBLE_XML.exists():
        subprocess.run(['git', 'clone', '--depth', '1', CORPUS_URL, str(RAW_BIBLE)], check=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in MULTILINGUAL_CORPUS + EVAL_ONLY_CORPUS:
        verses = extract_verses(BIBLE_XML / f'{name}.xml')
        write_json(PROCESSED / f'{name}.json', verses)
        print(f'{name}: {len(verses):,} verses')

    write_text_corpus(ENGLISH_CORPUS, PROCESSED / 'train_english.txt')
    write_text_corpus(MULTILINGUAL_CORPUS, PROCESSED / 'train_multilingual.txt')
    if args.no_tokenize:
        return

    from transformers import BertTokenizer

    for corpus, variant in [('english', 'bert'), ('multilingual', 'mbert')]:
        tokenizer = BertTokenizer.from_pretrained(VARIANTS[variant])
        tokenize_corpus(PROCESSED / f'train_{corpus}.txt', tokenizer, PROCESSED / f'tokenized_{corpus}')


if __name__ == '__main__':
    main()
