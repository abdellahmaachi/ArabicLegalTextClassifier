
from pathlib import Path


def load_documents(dataset_path):
    """
    Charge les documents .txt et leurs classes.

    Retourne un dictionnaire :
    {
        "classe/doc1.txt": {
            "text": "...",
            "class": "classe"
        }
    }
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dossier introuvable : {dataset_path}"
        )

    documents = {}

    # Parcourir les dossiers des classes
    for class_folder in sorted(dataset_path.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        # Parcourir les documents de chaque classe
        for file_path in sorted(class_folder.rglob("*.txt")):

            text = file_path.read_text(encoding="utf-8-sig")

            if not text.strip():
                continue

            # Identifiant unique du document
            doc_id = file_path.relative_to(dataset_path).as_posix()

            documents[doc_id] = {
                "text": text,
                "class": class_name
            }

    return documents


