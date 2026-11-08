
import re


def clean_text(text):
    """
    Nettoyage et normalisation du texte arabe.
    """

    # 1. Supprimer les diacritiques arabes
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # 2. Supprimer le Tatweel (ـ)
    text = text.replace("ـ", "")

    # 3. Normaliser les caractères arabes
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")

    # 4. Supprimer les chiffres et la ponctuation
    # Conserver uniquement les lettres arabes
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)

    # 5. Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Transforme un texte arabe en liste de mots.
    """

    cleaned_text = clean_text(text)

    tokens = cleaned_text.split()

    return tokens
