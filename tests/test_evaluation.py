
import unittest

from naive_bayes import build_document_map, train_naive_bayes
from evaluation import evaluate


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        documents = {
            "doc1": {
                "text": "عقار عقار ملكية",
                "class": "Immobilier"
            },
            "doc2": {
                "text": "زواج زواج ميراث",
                "class": "Successions"
            }
        }

        document_map = build_document_map(documents)
        self.model = train_naive_bayes(document_map)

    def test_perfect_accuracy(self):
        test_documents = {
            "test1": {
                "text": "عقار",
                "class": "Immobilier"
            },
            "test2": {
                "text": "زواج",
                "class": "Successions"
            }
        }

        results = evaluate(self.model, test_documents)

        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["macro_f1"], 1.0)

    def test_incorrect_prediction(self):
        test_documents = {
            "test1": {
                "text": "عقار",
                "class": "Successions"
            }
        }

        results = evaluate(self.model, test_documents)

        self.assertEqual(results["accuracy"], 0.0)
        self.assertEqual(
            results["confusion_matrix"]["Successions"]["Immobilier"],
            1
        )

    def test_empty_testset(self):
        with self.assertRaises(ValueError):
            evaluate(self.model, {})


if __name__ == "__main__":
    unittest.main()
