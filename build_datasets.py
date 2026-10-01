"""Build the semantic similarity and book classification datasets from the Bible corpus.

Run prepare_corpus.py first. Output goes to datasets/finetuning/{semantic_similarity,classification}.
"""
import random
from collections import defaultdict

from common import (BOOK_NAMES, FINETUNE_DATA, OT_BOOKS, SEED, load_verse_map,
                    split_80_10_10, write_json)

SIM_DIR = FINETUNE_DATA / 'semantic_similarity'
CLS_DIR = FINETUNE_DATA / 'classification'
VERSES_PER_BOOK = 150


def score(low, high):
    return round(random.uniform(low, high), 2)


def pair(s1, s2, sc, label, verse_id, pair_type):
    return {'sentence1': s1, 'sentence2': s2, 'score': sc, 'label': label,
            'verse_id': verse_id, 'pair_type': pair_type}


def group_by_book(kjv):
    by_book = defaultdict(list)
    for vid in kjv:
        parts = vid.split('.')
        if len(parts) >= 3 and parts[1] in BOOK_NAMES:
            by_book[parts[1]].append(vid)
    return by_book


def build_similarity(kjv, by_book):
    web, fra, spa, deu = (load_verse_map(n) for n in ['English-WEB', 'French', 'Spanish', 'German'])
    pairs = []

    # High: the same verse in two translations
    common_ids = sorted(set(kjv) & set(web) & set(fra) & set(spa) & set(deu))
    translation_pairs = [(kjv, web, 'KJV-WEB'), (kjv, fra, 'KJV-FRA'), (kjv, spa, 'KJV-SPA'),
                         (kjv, deu, 'KJV-DEU'), (web, fra, 'WEB-FRA')]
    for vid in random.sample(common_ids, min(3000, len(common_ids))):
        for t1, t2, name in translation_pairs:
            pairs.append(pair(t1[vid], t2[vid], score(0.85, 1.0), 'high', vid, name))

    # Medium: two verses from the same book
    for ids in by_book.values():
        if len(ids) < 20:
            continue
        sampled = random.sample(ids, min(60, len(ids)))
        for v1, v2 in zip(sampled[0::2], sampled[1::2]):
            pairs.append(pair(kjv[v1], kjv[v2], score(0.40, 0.70), 'medium', f'{v1}|{v2}', 'same-book'))

    # Low: Old Testament verse against New Testament verse
    ot_ids = [v for b in ['GEN', 'PSA', 'ISA', 'JER', 'EZE'] for v in by_book.get(b, [])]
    nt_ids = [v for b in ['MAT', 'JOH', 'ROM', 'REV'] for v in by_book.get(b, [])]
    for _ in range(4000):
        v1, v2 = random.choice(ot_ids), random.choice(nt_ids)
        pairs.append(pair(kjv[v1], kjv[v2], score(0.0, 0.25), 'low', f'{v1}|{v2}', 'cross-testament'))

    # Hard negatives: two verses from the same chapter, first 20 books only
    for book in list(by_book)[:20]:
        by_chapter = defaultdict(list)
        for vid in by_book[book]:
            by_chapter[vid.split('.')[2]].append(vid)
        for ids in by_chapter.values():
            if len(ids) < 4:
                continue
            sampled = random.sample(ids, 4)
            for v1, v2 in zip(sampled[0::2], sampled[1::2]):
                pairs.append(pair(kjv[v1], kjv[v2], score(0.20, 0.40), 'hard_negative',
                                  f'{v1}|{v2}', 'same-chapter'))

    random.shuffle(pairs)
    splits = split_80_10_10(pairs)
    for name, data in splits.items():
        write_json(SIM_DIR / f'{name}.json', data, indent=2)

    # Balanced training subset: downsample the dominant high and low classes
    train = splits['train']
    by_label = {label: [p for p in train if p['label'] == label]
                for label in ['high', 'medium', 'hard_negative', 'low']}
    target = max(len(by_label['medium']), len(by_label['hard_negative']), 2000)
    balanced = (random.sample(by_label['high'], min(target, len(by_label['high'])))
                + by_label['medium'] + by_label['hard_negative']
                + random.sample(by_label['low'], min(target, len(by_label['low']))))
    random.shuffle(balanced)
    write_json(SIM_DIR / 'train_balanced.json', balanced, indent=2)
    print(f'Similarity: {len(pairs):,} pairs, balanced train subset {len(balanced):,}')


def build_classification(kjv, by_book):
    samples = []
    for code, name in BOOK_NAMES.items():
        ids = by_book.get(code, [])
        for vid in random.sample(ids, min(VERSES_PER_BOOK, len(ids))):
            samples.append({'verse_id': vid, 'text': kjv[vid], 'book_code': code, 'book_name': name,
                            'testament': 'Old Testament' if code in OT_BOOKS else 'New Testament'})
    random.shuffle(samples)
    for name, data in split_80_10_10(samples).items():
        write_json(CLS_DIR / f'{name}.json', data, indent=2)

    books = sorted({s['book_name'] for s in samples})
    write_json(CLS_DIR / 'label_map.json', {name: i for i, name in enumerate(books)}, indent=2)
    print(f'Classification: {len(samples):,} samples, {len(books)} books')


def main():
    random.seed(SEED)
    kjv = load_verse_map('English')
    by_book = group_by_book(kjv)
    build_similarity(kjv, by_book)
    build_classification(kjv, by_book)


if __name__ == '__main__':
    main()
