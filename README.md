# Arabic Legal Text Classifier

**Arabic Legal Document Classification using Multinomial Naive Bayes From Scratch**

A Python-based Text Mining project that automatically classifies Arabic legal documents into two categories using a **Multinomial Naive Bayes classifier implemented from scratch**, without relying on machine learning libraries such as Scikit-learn.

The project covers Arabic text preprocessing, word frequency extraction, probability estimation with Laplace smoothing, document classification, and model evaluation.

## 1. Project Overview

The main objective is to build a supervised text classification system capable of identifying the legal category of a new Arabic document.

The dataset contains two classes:

- **الغرفة العقارية** — Real Estate Chamber
- **غرفة الأحوال الشخصية و الميراث** — Personal Status and Inheritance Chamber

The classifier learns word distributions from labeled training documents and uses Bayes' theorem to predict the most likely class of unseen texts.

## 2. Technologies

- **Language:** Python 3.11
- **Libraries:** Python standard library (`pathlib`, `re`, `collections`, `math`, `unittest`)
- **Algorithm:** Multinomial Naive Bayes
- **Text processing:** Arabic normalization and tokenization
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

No external machine learning dependencies are required.

## 3. Project Architecture

```text
ArabicLegalTextClassifier/
│
├── dataset/
│   ├── trainset/
│   │   ├── الغرفة العقارية/
│   │   └── غرفة الأحوال الشخصية و الميراث/
│   └── testset/
│       ├── الغرفة العقارية/
│       └── غرفة الأحوال الشخصية و الميراث/
│
├── tests/
│   ├── __init__.py
│   ├── test_loader_preprocessing.py
│   ├── test_naive_bayes.py
│   └── test_evaluation.py
│
├── loader.py
├── preprocessing.py
├── naive_bayes.py
├── evaluation.py
├── main.py
├── classify.py
├── .gitignore
└── README.md
```

### Module Responsibilities

| Module | Description |
|---|---|
| `loader.py` | Loads documents and their class labels from the dataset |
| `preprocessing.py` | Performs Arabic text cleaning, normalization, and tokenization |
| `naive_bayes.py` | Implements word frequency maps, probability estimation, and prediction |
| `evaluation.py` | Computes classification metrics and the confusion matrix |
| `main.py` | Runs training, evaluation, and example classification |
| `classify.py` | Provides interactive text and `.txt` document classification |
| `tests/` | Contains unit tests for the project modules |

## 4. Methodology

### Step 1 — Document Loading

Documents are loaded from the training and testing folders. Each document receives a class label based on its parent directory.

### Step 2 — Arabic Text Preprocessing

The preprocessing pipeline includes:

1. Removal of Arabic diacritics and Tatweel.
2. Normalization of selected Arabic characters.
3. Removal of punctuation, numbers, and non-Arabic characters.
4. Tokenization into individual Arabic words.

The same preprocessing functions are applied during training and prediction.

### Step 3 — Word Frequency Map

The first dictionary stores word occurrences for each document and class.

**Data structure:**

`Map<Document, Map<(Word, Class), Frequency>>`

Example:

```python
{
    "doc1.txt": {
        ("عقار", "الغرفة العقارية"): 2,
        ("ملكية", "الغرفة العقارية"): 1
    }
}
```

### Step 4 — Naive Bayes Training

Word probabilities are calculated using **Laplace (Add-One) Smoothing**:

$$\[
P(w_k \mid C_i) = \frac{n_k + 1}{n_i + m}
\]$$

Where:

- `nk`: Number of occurrences of word `wk` in class `Ci`.
- `ni`: Total number of word occurrences in class `Ci`.
- `m`: Total number of unique words in the training vocabulary.

Class prior probabilities are calculated as:

$$\[
P(C_i)=\frac{N_i}{N}
\]$$

Where `Ni` is the number of training documents belonging to class `Ci`, and `N` is the total number of training documents.

**Probability map:**

`Map<Class, Map<Word, Probability>>`

This dictionary stores the conditional probability of each vocabulary word for each class.

### Step 5 — Document Classification

For a new document, the classifier computes a score for each class using log probabilities:

$$\[
Score(C_i,d)=\log P(C_i)+\sum_{w\in V_d}f(w,d)\log P(w\mid C_i)
\]$$

The predicted class is the one with the highest score:

$$\[
C_{predicted}=\arg\max_{C_i} Score(C_i,d)
\]$$

Log probabilities are used to avoid numerical underflow when processing long documents.

Words absent from the training vocabulary are ignored during prediction.

### Step 6 — Model Evaluation

The model is evaluated on a separate test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix

All evaluation metrics are implemented without Scikit-learn.

## 5. Experimental Results

The following results were obtained from the current dataset.

### Dataset Statistics

| Property | Value |
|---|---:|
| Training documents | 2,222 |
| Testing documents | 19 |
| Vocabulary size | 49,878 |
| Number of classes | 2 |

### Classification Performance

| Metric | Result |
|---|---:|
| Accuracy | 100.00% |
| Macro F1-score | 100.00% |
| Precision — both classes | 100.00% |
| Recall — both classes | 100.00% |
| F1-score — both classes | 100.00% |

### Confusion Matrix

| Actual / Predicted | Real Estate | Personal Status |
|---|---:|---:|
| Real Estate | 10 | 0 |
| Personal Status | 0 | 9 |

**Note:** These results were obtained using only 19 test documents. Although all test documents were correctly classified, additional evaluation on a larger independent test set would be necessary to assess generalization reliably.

## 6. Installation and Usage

### Requirements

- Python 3.11 or a compatible Python 3 version
- Git

### Clone the Repository

```bash
git clone https://github.com/abdellahmaachi/ArabicLegalTextClassifier.git
cd ArabicLegalTextClassifier
```

### Prepare the Dataset

Create the following folders if they do not already exist:

```text
dataset/
├── trainset/
│   ├── الغرفة العقارية/
│   └── غرفة الأحوال الشخصية و الميراث/
└── testset/
    ├── الغرفة العقارية/
    └── غرفة الأحوال الشخصية و الميراث/
```

Place the corresponding UTF-8 `.txt` files in each class folder.

**The dataset is not included in this GitHub repository.** Users must provide an appropriately labeled dataset before running training and evaluation.

### Run Training and Evaluation

```bash
python main.py
```

This command loads the data, trains the Naive Bayes model, evaluates it, and performs an example prediction.

### Classify a New Document

```bash
python classify.py
```

Choose between:

1. Entering Arabic text manually.
2. Providing the path to an existing `.txt` file.

The application displays the predicted legal category.

### Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## 7. Project Scope and Limitations

- The classifier supports the two legal categories included in the training dataset.
- The algorithm uses a bag-of-words representation and does not model word order or semantic context.
- Unknown words are ignored during prediction.
- The current evaluation set contains only 19 documents.
- The trained model is rebuilt from the training dataset whenever the application starts.


**Academic Project — Text Mining | Python | Machine Learning | Natural Language Processing**
