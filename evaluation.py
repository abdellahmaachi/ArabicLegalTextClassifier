
from naive_bayes import predict


def evaluate(model, test_documents):
    """
    Évalue le modèle sur les documents de test.
    Calcule Accuracy, Precision, Recall et F1-score.
    """

    classes = list(model["prior_map"].keys())

    if not test_documents:
        raise ValueError("Aucun document dans le testset.")

    # Matrice de confusion : classe réelle -> classe prédite
    confusion = {
        actual: {predicted: 0 for predicted in classes}
        for actual in classes
    }

    # 1. Prédiction des documents de test
    for doc_id, data in test_documents.items():
        actual = data["class"]

        if actual not in confusion:
            raise ValueError(f"Classe inconnue : {actual}")

        predicted = predict(data["text"], model)

        confusion[actual][predicted] += 1

    # 2. Calcul de l'Accuracy
    total = len(test_documents)
    correct = sum(confusion[c][c] for c in classes)

    accuracy = correct / total

    # 3. Precision, Recall et F1-score par classe
    metrics = {}

    for classe in classes:
        tp = confusion[classe][classe]

        fp = sum(
            confusion[c][classe]
            for c in classes if c != classe
        )

        fn = sum(
            confusion[classe][c]
            for c in classes if c != classe
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        metrics[classe] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # 4. Moyenne macro du F1-score
    macro_f1 = sum(
        metrics[c]["f1"] for c in classes
    ) / len(classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "metrics": metrics,
        "confusion_matrix": confusion
    }
