
import unittest

from naive_bayes import (
    build_document_map,
    train_naive_bayes,
    predict
)


class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        self.documents = {
            "doc1.txt": {
                "text": "عقار ملكية عقار",
                "class": "Immobilier"
            },
            "doc2.txt": {
                "text": "زواج ميراث زواج",
                "class": "Successions"
            }
        }

        self.document_map = build_document_map(self.documents)
        self.model = train_naive_bayes(self.document_map)

    def test_document_map(self):
        self.assertEqual(
            self.document_map["doc1.txt"][("عقار", "Immobilier")],
            2
        )

    def test_prior_probability(self):
        self.assertEqual(
            self.model["prior_map"]["Immobilier"],
            0.5
        )

    def test_laplace_probability(self):
        probability = self.model["probability_map"]["Immobilier"]["عقار"]

        self.assertAlmostEqual(probability, 3 / 7)

    def test_prediction(self):
        result = predict("عقار عقار", self.model)

        self.assertEqual(result, "Immobilier")


if __name__ == "__main__":
    unittest.main()
