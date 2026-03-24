import json
from rapidfuzz import process


class MedicineMatcher:
    def __init__(self, vocab_path="medicine_vocab.json"):
        # Load vocabulary
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

    def match(self, text, top_k=3):
        text = text.lower().strip()

        # Get best matches
        matches = process.extract(
            text,
            self.vocab,
            limit=top_k
        )

        # Format output
        results = []
        for match in matches:
            results.append({
                "medicine": match[0],
                "score": match[1]
            })

        return results

    def best_match(self, text):
        matches = self.match(text, top_k=1)

        if matches:
            return matches[0]["medicine"], matches[0]["score"]
        else:
            return text, 0