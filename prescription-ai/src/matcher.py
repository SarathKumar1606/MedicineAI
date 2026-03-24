import json

def load_vocab(vocab_path="medicine_vocab.json"):
    with open(vocab_path, "r") as f:
        return json.load(f)


def match_medicines(text, vocab_path="medicine_vocab.json"):
    vocab = load_vocab(vocab_path)

    found = []
    text_lower = text.lower()

    for med in vocab:
        if med.lower() in text_lower:
            found.append(med)

    return found