from loader import load_documents
from naive_bayes import (
    build_document_map,
    train_naive_bayes,
    predict
)

# 1. Charger les documents
documents = load_documents("dataset/trainset")

# 2. Construire Map 1
document_map = build_document_map(documents)

# 3. Entraîner Naive Bayes et construire Map 2
model = train_naive_bayes(document_map)

# 4. Afficher quelques informations
print("Documents :", len(document_map))
print("Vocabulaire :", len(model["vocabulary"]))
print("Classes :", list(model["prior_map"]))

# 5. Classifier un nouveau texte
texte = "عقد بيع عقار وملكية أرض"

classe = predict(texte, model)

print("Classe prédite :", classe)