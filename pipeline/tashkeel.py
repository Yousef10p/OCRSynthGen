"""Tashkeel: Arabic diacritics (harakat) on the slot texts, so the OCR model also learns marked text.

Wikipedia text has almost no marks, so they are added by a small diacritization model: text2tashkeel's
default "rawi-ensemble" (BiLSTM models in ONNX, run by onnxruntime, offline, about 2% diacritic error rate
as its authors report; https://github.com/TigreGotico/text2tashkeel, Apache 2.0). The text itself never
changes: a word takes the model's marks only if its letters are the same as before (the model also "restores"
hamza, sometimes wrongly: لامائي -> لَأَمَائِيٍّ), and a word that already had marks in Wikipedia keeps them.
Latin letters, digits and punctuation pass through unchanged.

Per page, one level:
    none      the text as it is (most printed Arabic)
    partial   shadda kept everywhere and all marks on about a third of the words (how newspapers, school
              books and signs often mark only some words)
    full      every letter marked (religious, classical and teaching texts)

The labels are always what is drawn: the ground truth comes from the rendered page's words, marks included.
"""
import re

LEVELS = {"none": 0.4, "partial": 0.3, "full": 0.3}
ARABIC = re.compile(r"[ء-ي]")
MARKS = re.compile(r"[ً-ْٰ]")
VOWELS = re.compile(r"[ً-ِْٰ]")  # every mark but shadda (U+0651)
PARTIAL_WORDS = 0.35
LINE_HEIGHT = {"partial": 1.6, "full": 1.75}  # the least line height a page with marks gets
_model = None


def model():
    global _model
    if _model is None:
        from text2tashkeel import Diacritizer
        _model = Diacritizer()  # one onnxruntime session, reused
    return _model


def apply(text, level, rng):
    """The text with marks at this level ("none", "partial" or "full")."""
    if level == "none" or not ARABIC.search(text):
        return text
    words, marked = text.split(" "), model().diacritize(text).split(" ")
    if len(marked) != len(words):  # can't pair the words up: leave the text as it is
        return text
    words = [m if MARKS.sub("", m) == w else w for w, m in zip(words, marked)]  # same letters only
    if level == "full":
        return " ".join(words)
    return " ".join(w if rng.random() < PARTIAL_WORDS else VOWELS.sub("", w) for w in words)
