
from loader import load_documents
from naive_bayes import (
    build_document_map,
    train_naive_bayes,
    predict
)
from evaluation import evaluate


# 1. Chargement des documents
train_documents = load_documents("dataset/trainset")
test_documents = load_documents("dataset/testset")

# 2. Construction de Map 1
document_map = build_document_map(train_documents)

# 3. Entraînement et construction de Map 2
model = train_naive_bayes(document_map)

print("Documents train :", len(train_documents))
print("Documents test :", len(test_documents))
print("Vocabulaire :", len(model["vocabulary"]))

# 4. Évaluation du modèle
results = evaluate(model, test_documents)

print("\n===== RÉSULTATS =====")
print(f"Accuracy : {results['accuracy']:.2%}")
print(f"Macro F1 : {results['macro_f1']:.2%}")

for classe, metrics in results["metrics"].items():
    print(f"\nClasse : {classe}")
    print(f"Precision : {metrics['precision']:.2%}")
    print(f"Recall    : {metrics['recall']:.2%}")
    print(f"F1-score  : {metrics['f1']:.2%}")

print("\n===== MATRICE DE CONFUSION =====")

for classe, predictions in results["confusion_matrix"].items():
    print(classe, ":", predictions)

# 5. Classification d'un nouveau document
text = "عقد بيع عقار وملكية أرض"

prediction = predict(text, model)

print("\nNouveau texte :", text)
print("Classe prédite :", prediction)
