"""Generate the extractive QA dataset with Llama 3.3 70B through the Groq API.

    export GROQ_API_KEY=...
    python generate_qa.py

Progress is checkpointed, so the script can be stopped and rerun.
"""
import json
import os
import random
import time

from groq import Groq

from common import BOOK_NAMES, FINETUNE_DATA, OT_BOOKS, SEED, book_code, load_verse_map, read_json, split_80_10_10, write_json

QA_DIR = FINETUNE_DATA / 'qa'
CHECKPOINT = QA_DIR / 'qa_checkpoint.json'
MODEL = 'llama-3.3-70b-versatile'
VERSES_PER_TESTAMENT = 600

SYSTEM_PROMPT = """You are an expert annotator creating a question answering dataset from Bible verses.
Generate one factual question-answer pair. The answer MUST be a direct, continuous substring of the verse.
Return ONLY valid JSON with keys: context, question, answer"""
USER_TEMPLATE = 'Verse: "{verse}"\n\nGenerate one question-answer pair where the answer is a direct substring of the verse.'


def is_suitable(text):
    """60 to 300 characters with at least two capitalised words after the first (names, places)."""
    if not 60 <= len(text) <= 300:
        return False
    return sum(1 for w in text.split()[1:] if w[0].isupper() and w.isalpha()) >= 2


def select_verses(kjv):
    pools = {'Old Testament': [], 'New Testament': []}
    for vid, text in kjv.items():
        code = book_code(vid)
        if code in BOOK_NAMES and is_suitable(text):
            testament = 'Old Testament' if code in OT_BOOKS else 'New Testament'
            pools[testament].append({'verse_id': vid, 'text': text, 'book_name': BOOK_NAMES[code],
                                     'testament': testament})
    selected = []
    for pool in pools.values():
        selected += random.sample(pool, min(VERSES_PER_TESTAMENT, len(pool)))
    random.shuffle(selected)
    return selected


def locate_answer(verse, answer):
    """Return (answer, start) using the verse's own casing, or None if it is not a substring."""
    start = verse.find(answer)
    if start == -1:
        start = verse.lower().find(answer.lower())
        if start == -1:
            return None
        answer = verse[start:start + len(answer)]
    return answer, start


def main():
    random.seed(SEED)
    verses = select_verses(load_verse_map('English'))
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    dataset = read_json(CHECKPOINT) if CHECKPOINT.exists() else []
    start_index = max((int(d['id'].split('-')[-1]) for d in dataset), default=-1) + 1

    for i in range(start_index, len(verses)):
        entry, text = verses[i], verses[i]['text']
        try:
            response = client.chat.completions.create(
                model=MODEL, temperature=0.3, max_tokens=200, response_format={'type': 'json_object'},
                messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                          {'role': 'user', 'content': USER_TEMPLATE.format(verse=text)}],
            )
            parsed = json.loads(response.choices[0].message.content.strip())
            question, answer = parsed.get('question', '').strip(), parsed.get('answer', '').strip()
            found = locate_answer(text, answer) if question and answer else None
            if found:
                answer, start = found
                dataset.append({
                    'id': f'religionbert-qa-{i:04d}', 'verse_id': entry['verse_id'],
                    'book_name': entry['book_name'], 'testament': entry['testament'],
                    'context': text, 'question': question, 'answer': answer, 'answer_start': start,
                    'answers': {'text': [answer], 'answer_start': [start]},
                })
                if len(dataset) % 50 == 0:
                    write_json(CHECKPOINT, dataset, indent=2)
            time.sleep(3.0)  # stay under the Groq daily token limit
        except Exception as e:
            print(f'Error at {i}: {e}')
            time.sleep(10)

    random.shuffle(dataset)
    for name, data in split_80_10_10(dataset).items():
        write_json(QA_DIR / f'{name}.json', data, indent=2)
    print(f'QA dataset: {len(dataset)} examples')


if __name__ == '__main__':
    main()
