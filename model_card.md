# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I primarily used the rule based model in `mood_analyzer.py`. I did not tune the ML model in this iteration.

**Intended purpose:**  
Classify short informal text posts into `positive`, `negative`, `neutral`, or `mixed` mood labels.

**How it works (brief):**  
The rule based model tokenizes text, assigns +1 to positive words and -1 to negative words, applies a negation rule (for example `not happy`), and then maps the final score to a label. It also includes a small sarcasm heuristic for `stuck in traffic` when positive wording appears.



## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` currently has 14 posts. I extended the starter set by adding 3 examples:
- `I have no strong feelings one way or the other about this` (`neutral`)
- `I'm so tired but also kind of excited for tomorrow's event` (`mixed`)
- `Just got a promotion at work, but now I have to deal with more stress 😩` (`mixed`)

**Labeling process:**  
Labels were assigned by dominant or balanced sentiment signals:
- `neutral` when there is no clear positive/negative cue.
- `mixed` when both positive and negative feelings are clearly present.
Hard/ambiguous cases included sarcasm (`I absolutely love getting stuck in traffic`) and contrastive mood posts using `but`.

**Important characteristics of your dataset:**  
- Contains slang (`no cap`, `lowkey`, `highkey`)
- Contains emoji (`😂`, `😩`)
- Includes sarcasm and contrastive language (`but`)
- Includes short ambiguous text (`This is fine`)
- Includes mixed-feeling statements

**Possible issues with the dataset:**  
- Small dataset size (14 total examples)
- Some labels are subjective (`mixed` vs `neutral`)
- Limited coverage of slang variants and emoji meanings
- No dedicated train/validation/test split for robust generalization estimates

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
- **Preprocessing:** lowercase, punctuation filtering via regex tokenization, keep contractions/emoticons/selected emojis as tokens.
- **Base score:** +1 for tokens in `POSITIVE_WORDS`, -1 for tokens in `NEGATIVE_WORDS`.
- **Negation enhancement:** if a negation token (for example `not`, `never`, `can't`) appears immediately before a sentiment word, flip that word's contribution.
- **Sarcasm enhancement:** if positive wording appears with `stuck in traffic` / `in traffic`, apply an extra negative penalty.
- **Label mapping:** score > 0 => `positive`, score < 0 => `negative`, score == 0 => `mixed` only if both positive and negative lexicon hits exist, otherwise `neutral`.
- **Lexicon tuning:** added `hopeful` to `POSITIVE_WORDS` to fix a repeated mixed-sentiment miss.

**Strengths of this approach:**  
- Transparent and easy to debug (you can inspect tokens and score directly)
- Good on direct sentiment text and simple negation (`not happy`, `not bad`)
- Fast to iterate by adjusting words/rules for known errors

**Weaknesses of this approach:**  
- Brittle outside known keywords and handcrafted rules
- Still weak on subtle sarcasm and context (humor can invert literal sentiment)
- Can miss mixed mood when one side uses unseen words (for example `proud` not in lexicon)
- Emoji nuance is only partially handled

## 4. How the ML Model Works (if used)

**Features used:**  
ML baseline in `ml_experiments.py` uses bag-of-words features via `CountVectorizer`.

**Training data:**  
The ML script trains on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**  
I did not run a full ML comparison in this pass, so no new ML tuning/accuracy claims are included here.

**Strengths and weaknesses:**  
Potential strengths: learns correlations automatically without hand-writing rules. Potential weaknesses: overfits tiny data and may learn spurious token-label shortcuts.

## 5. Evaluation

**How you evaluated the model:**  
I evaluated the rule based model using `evaluate_rule_based(SAMPLE_POSTS, TRUE_LABELS)` in `main.py`.

Observed rule based accuracy: **0.79** on the current 14-post labeled dataset.

**Examples of correct predictions:**  
- `I am not happy about this` -> predicted `negative` (negation + positive word flips correctly)
- `I can't decide if I'm happy or sad about this news` -> predicted `mixed` (balanced positive + negative cues)
- `I absolutely love getting stuck in traffic` -> predicted `negative` (sarcasm heuristic catches traffic frustration phrase)

**Examples of incorrect predictions:**  
- `Lowkey stressed but highkey proud of myself` -> predicted `negative`, true `mixed` (captures `stressed` but misses positive cue `proud`)
- `This movie was so bad it was actually hilarious 😂` -> predicted `negative`, true `positive` (literal `bad` dominates; humor/emoji inversion not modeled)
- `Just got a promotion at work, but now I have to deal with more stress 😩` -> predicted `neutral`, true `mixed` (positive event `promotion` not in lexicon; `stress` may not match `stressed`)

These three are the sentences that consistently confused the model in current evaluation.

## 6. Limitations

- Very small and hand-labeled dataset
- Heavy dependence on lexicon coverage and exact word forms
- Limited context understanding beyond local rules
- Sarcasm and humor handling remains narrow and pattern-specific
- Accuracy is measured on the same small labeled set, not an external test set

## 7. Ethical Considerations

- Misclassification risk: a distress message could be incorrectly labeled as neutral/positive.
- Language fairness risk: slang/dialect not represented in the lexicon can be misread.
- Context loss risk: literal words may conflict with true intent (sarcasm, jokes, cultural references).
- Privacy risk: mood analysis on personal text can expose sensitive emotional information.

**Bias and scope note:**  
This model is currently optimized for short, English, social-media-style text similar to the examples in `SAMPLE_POSTS` (including terms like `no cap`, `lowkey`, and simple emoji usage). It may misinterpret users who write in other dialects, code-switch between languages, use region-specific slang not covered by the lexicon, or rely on cultural references and humor patterns not represented in the dataset.

## 8. Ideas for Improvement

- Expand lexicons with missing positive/negative variants (for example `proud`, `stress`, `hilarious`) and slang mappings.
- Add a small rule for contrastive conjunctions (`but`) to better detect `mixed` mood.
- Improve emoji and sarcasm handling beyond one phrase trigger.
- Evaluate ML baseline with TF-IDF and compare against rule based errors.
- Create a held-out test split (or cross-validation) rather than relying on same-set accuracy.
- Add error analysis logging (`tokens`, `score`, triggered rules) for faster iterative debugging.

## Based on the ML extension in ml_experiments.py:
The model learned quickly and did not require any adjustments or additional test cases to correctly categorize the sample_posts. Did not struggle with sarcasm or mixed posts at all. It was much more sensitive. It did not introduce any new issues. 
