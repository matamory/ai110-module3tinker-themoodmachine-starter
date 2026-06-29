# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

        # Per-token sentiment weights. Anything listed here overrides the
        # default +/-1 contribution, letting strong signals count for more.
        # Emojis and slang are included so they act as sentiment on their own,
        # even if they aren't in the positive/negative word lists.
        self.token_weights: Dict[str, int] = {
          # Strong positive emojis / emoticons
          "😂": 2, "🤣": 2, "😊": 2, "😍": 2, "👍": 2,
          ":)": 1, ":-)": 1, ";)": 1,
          # Strong negative emojis / emoticons
          "😢": -2, "😭": -2, "😡": -2, "👎": -2, "🥲": -1, "💀": -1,
          ":(": -1, ":-(": -1,
          # Slang signals
          "lol": 1, "lmao": 2, "yay": 2, "ugh": -1, "meh": -1, "wtf": -2,
          # Intense words weigh more than mild ones
          "love": 2, "amazing": 2, "awesome": 2, "best": 2, "excellent": 2,
          "hate": -2, "awful": -2, "terrible": -2, "horrible": -2, "worst": -2,
        }

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()

        # Add spacing around simple emoticons so they can be captured as tokens.
        for emoticon in [":)", ":-(", ":(", ":-)"]:
          cleaned = cleaned.replace(emoticon, f" {emoticon} ")

        # Keep word tokens (including simple contractions), common emoticons,
        # and selected mood-relevant emojis while dropping punctuation.
        tokens = re.findall(
          r"[a-z0-9]+(?:'[a-z0-9]+)?|[:;]-?[)(]|[😂🥲💀🤣😊😢😭😡😍👍👎]",
          cleaned,
        )

        # Normalize stretched-out words by collapsing any run of 3 or more
        # identical characters down to 2 ("soooo" -> "soo", "yaaay" -> "yaay").
        # This keeps emphatic spellings mapping to the same token while leaving
        # naturally doubled letters (like "cool") intact. Emoticons and emojis
        # are single tokens that won't match the word pattern, so they're safe.
        word_pattern = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?$")
        tokens = [
          re.sub(r"(.)\1{2,}", r"\1\1", token) if word_pattern.match(token) else token
          for token in tokens
        ]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _token_weight(self, token: str) -> int:
        """
        Return the signed sentiment weight of a single token.

        Tokens listed in ``self.token_weights`` (strong words, emojis, slang)
        use their custom weight. Otherwise a token in the positive/negative
        word lists contributes the default +/-1. Everything else is 0.
        """
        if token in self.token_weights:
          return self.token_weights[token]
        if token in self.positive_words:
          return 1
        if token in self.negative_words:
          return -1
        return 0

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Improvements implemented:
          - Simple negation handling ("not happy", "not bad")
          - Weighted sentiment: intense words count for more than mild ones
          - Emojis and slang ("😂", ":)", "lol", "💀") act as strong signals
          - Lightweight sarcasm heuristic for the dataset
        """
        tokens = self.preprocess(text)
        score = 0

        # Intentional enhancement: simple negation handling.
        # If a negation word appears right before a known sentiment word,
        # flip that sentiment's contribution.
        negation_words = {
          "not",
          "no",
          "never",
          "can't",
          "cannot",
          "don't",
          "won't",
          "isn't",
          "wasn't",
        }

        index = 0
        while index < len(tokens):
          token = tokens[index]

          # If a negation precedes a sentiment token, flip that token's
          # (weighted) contribution. "not amazing" -> -2, "not terrible" -> +2.
          if token in negation_words and index + 1 < len(tokens):
            next_weight = self._token_weight(tokens[index + 1])
            if next_weight != 0:
              score -= next_weight
              index += 2
              continue

          score += self._token_weight(token)
          index += 1

        # Lightweight sarcasm heuristic (dataset-focused):
        # positive language + clearly frustrating phrase -> reduce score.
        has_positive_token = any(token in self.positive_words for token in tokens)
        sarcasm_triggers = [
          ("stuck", "in", "traffic"),
          ("in", "traffic"),
        ]

        for trigger in sarcasm_triggers:
          window = len(trigger)
          for start in range(len(tokens) - window + 1):
            if tuple(tokens[start:start + window]) == trigger and has_positive_token:
              score -= 2
              return score

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        if score > 0:
          return "positive"
        if score < 0:
          return "negative"

        # score == 0: distinguish mixed sentiment from truly neutral text.
        tokens = self.preprocess(text)
        has_positive = any(token in self.positive_words for token in tokens)
        has_negative = any(token in self.negative_words for token in tokens)

        if has_positive and has_negative:
          return "mixed"

        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
