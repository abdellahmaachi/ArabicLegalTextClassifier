from collections import Counter
from preprocessing import tokenize


def build_document_map(documents):
    document_map = {}

    for doc_id, data in documents.items():
        classe = data["class"]
        tokens = tokenize(data["text"])

        occurrences = Counter(tokens)

        if not occurrences:
            continue

        document_map[doc_id] = {
            (mot, classe): nb
            for mot, nb in occurrences.items()
        }

    return document_map

from collections import Counter, defaultdict


def train_naive_bayes(document_map):
    if not document_map:
        raise ValueError("Aucun document pour l'entraînement.")

    class_doc_counts = Counter()
    class_word_counts = defaultdict(Counter)
    vocabulary = set()

    # Compter les mots par classe
    for doc_words in document_map.values():

        classe = next(iter(doc_words))[1]
        class_doc_counts[classe] += 1

        for (mot, _), nb in doc_words.items():
            class_word_counts[classe][mot] += nb
            vocabulary.add(mot)

    # Probabilités a priori
    total_docs = sum(class_doc_counts.values())

    prior_map = {
        classe: nb / total_docs
        for classe, nb in class_doc_counts.items()
    }

    # Probabilités conditionnelles
    m = len(vocabulary)
    probability_map = {}

    for classe, counts in class_word_counts.items():
        n = sum(counts.values())

        probability_map[classe] = {
            mot: (counts[mot] + 1) / (n + m)
            for mot in vocabulary
        }

    return {
        "prior_map": prior_map,
        "probability_map": probability_map,
        "vocabulary": vocabulary
    }

import math
from collections import Counter
from preprocessing import tokenize


def predict(text, model):
    tokens = tokenize(text)
    occurrences = Counter(tokens)

    scores = {}

    for classe, prior in model["prior_map"].items():
        score = math.log(prior)

        for mot, nb in occurrences.items():

            if mot in model["vocabulary"]:
                probability = model["probability_map"][classe][mot]

                score += nb * math.log(probability)

        scores[classe] = score

    return max(scores, key=scores.get)


