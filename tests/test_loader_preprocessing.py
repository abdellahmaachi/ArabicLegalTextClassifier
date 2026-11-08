
import unittest
import tempfile
from pathlib import Path

from loader import load_documents
from preprocessing import clean_text, tokenize


class TestLoaderPreprocessing(unittest.TestCase):

    # Test 1 : Chargement des documents
    def test_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "الغرفة العقارية"
            folder.mkdir()

            (folder / "doc1.txt").write_text(
                "عقد بيع عقار", encoding="utf-8"
            )

            documents = load_documents(temp_dir)

            self.assertEqual(len(documents), 1)

            doc = documents["الغرفة العقارية/doc1.txt"]
            self.assertEqual(doc["class"], "الغرفة العقارية")
            self.assertEqual(doc["text"], "عقد بيع عقار")

    # Test 2 : Dossier inexistant
    def test_missing_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "inexistant"

            with self.assertRaises(FileNotFoundError):
                load_documents(missing)

    # Test 3 : Nettoyage du texte arabe
    def test_clean_text(self):
        text = "إِنَّ العَقَارَ والمِلْكِيَّةَ، 2026!"

        result = clean_text(text)

        self.assertEqual(
            result, "ان العقار والملكية"
        )

    # Test 4 : Tokenisation
    def test_tokenize(self):
        text = "عقد بيع عقار"

        result = tokenize(text)

        self.assertEqual(
            result, ["عقد", "بيع", "عقار"]
        )

    def test_real_dataset(self):
        train_path = Path("dataset/trainset")

        if not train_path.is_dir():
            self.skipTest("Dataset réel indisponible")

        documents = load_documents(train_path)

        self.assertGreater(len(documents), 0)

        for doc_id, data in documents.items():
            self.assertTrue(data["text"].strip())
            self.assertTrue(data["class"])


if __name__ == "__main__":
    unittest.main()
