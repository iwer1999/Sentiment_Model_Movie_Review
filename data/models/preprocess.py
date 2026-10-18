"""Text cleaning used by BOTH training and the app, so they always match."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that change meaning ("not good") or strength ("very good") must be kept,
# so we remove them from the stop-word list.
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "neither",
    "nowhere", "cannot", "without", "very", "too", "more", "most",
    "less", "least", "enough",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS


def clean_text(text: str) -> str:
    # 1. Consistent case: "Great" and "great" become the same word.
    text = text.lower()

    # 2. Remove HTML tags such as <br /> (IMDb reviews contain many).
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Expand contractions so negation survives: "isn't good" -> "is not good".
    text = text.replace("can't", "cannot").replace("won't", "will not")
    text = text.replace("n't", " not")

    # 4. Remove punctuation, digits and symbols; keep only letters and spaces.
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Tokenization: split into words on whitespace.
    tokens = text.split()

    # 6. Remove stop words (e.g. "the", "is") and one-letter leftovers.
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


if __name__ == "__main__":
    demo = "This movie wasn't good at all!<br /><br />The acting was NOT great."
    print("Before:", demo)
    print("After: ", clean_text(demo))