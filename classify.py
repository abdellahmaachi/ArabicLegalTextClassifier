
from pathlib import Path
from loader import load_documents
from naive_bayes import (
    build_document_map,
    train_naive_bayes,
    predict
)


def main():
    # Charger et entraîner le modèle existant
    documents = load_documents("dataset/trainset")
    document_map = build_document_map(documents)
    model = train_naive_bayes(document_map)

    print("\n=== Arabic Legal Text Classifier ===")
    print("1. Saisir un texte arabe")
    print("2. Classifier un fichier .txt")

    choix = input("Votre choix : ").strip()

    if choix == "1":
        text = input("Entrez votre texte : ")

    elif choix == "2":
        chemin = input("Chemin du fichier : ").strip().strip('"')
        text = Path(chemin).read_text(encoding="utf-8-sig")

    else:
        print("Choix invalide.")
        return

    if not text.strip():
        print("Le texte est vide.")
        return

    classe = predict(text, model)

    print("\nClasse prédite :", classe)


if __name__ == "__main__":
    main()
