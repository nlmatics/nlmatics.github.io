---
layout: page
title: "Boolean Question Answering with Neutral Labels"
date: 2020-3-12 10:30:00 -0000
author: Evan Li
image: site_files/Placeholder-thumb.png

---

# Boolean Question Answering with Neutral Labels

By Evan Li

## Introduction

![yesno](/site_files/evan-post-imgs/yesno.jpeg)

"If Lionel Messi, an Argentinean soccer player, transfers to Man City from Barcelona, it would be his first transfer in his club career."

Based on this sentence, humans arguably have enough information to answer these Yes/No questions and more:

1) Is Messi a soccer player? - Yes

2) Is Messi American? - No

2) is Barcelona a soccer club? - Yes

3) is Man City a soccer club? - Yes

4) has Messi played for Man City before? - No 

5) has Messi at least played for one club? - Yes

6) Lionel Messi previously transferred from Sevilla to Barcelona - No

A single sentence can provide answers to a many Yes/No questions. Some of these questions are obvious, while others take reasoning to answer. 

Given a passage and boolean question, transformer Q&A models can be trained to predict whether the answer is Yes/No. However, what if someone asks the question "Is Messi better than Ronaldo?". The sentence above does not contain enough information to answer the question. In that case, the the model should predict something along the lines of "I don't know". Single sentences are only able to answer a tiny subset of Yes/No questions, so it is quite embarrassing for a model to always be confident it has an answer.

However, there is a problem: current datasets for boolean question answering only contain Yes/No labels. Is there a way to train a boolean QA model to be able to predict neutral without having additionally hand-annotate thousands of neutral samples?

## Our Approach

Boolean question answering is very similar to another popular NLP task: textual entailment. The textual entailment task takes a passage and a claim and predicts whether the claim is entailed, contradicted, or neutral to the passage. 

For example, 

Passage: If Lionel Messi, an Argentinean soccer player, transfers to Man City from Barcelona, it would be his first transfer in his club career.

Claim: Messi is a soccer player

Then the passage would entail the claim.

If we convert all boolean questions to claims (e.g. "is Messia a soccer player?" —> "Messi is a soccer player"), then boolean Q&A would be the exact same task as textual entailment. This means we would be able to steal neutral samples from existing entailment datasets to train our Boolean QA model.

## Question to Phrase Conversion

We use Spacy NER to convert boolean questions to phrases. [Code can be found here](https://github.com/nlmatics/nlmatics.github.io/blob/gh-pages/docs/code-examples/boolq_to_phrases.py). Here is a summary of the algorithm:

1) Find the boolean question word. Usually the first word is a boolean question word. If it isn't, then we scan the rest of the question for the boolean question word.

2) Find the noun subject of the question. Place the boolean question word after the noun subject, and we are done

3) if a noun subject is not found, then place the question word in between the first noun followed by a verb or adjective.

4) If nothing is found in step 3, we place the question word after the first noun that we find.

_Examples of the question to phrase conversions_

| Original Question                                            | New Phrase                                                   | Comment                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| “is there any litigation filed against Virginia Public Schools?” | “there is litigation filed against virginia public schools.” | We move the question word in between a pronoun and a verb  |
| “are Robert and George both acting in the school play?” | “robert and george are both acting in the school play.”       | Sometimes there were multiple noun subjects - in this case, we moved the question word after the latest continuous string of noun subjects |
| “if you get vaccinated, will you be going back to school in the summer?”                      | “if you get vaccinated, you will be going back to school in the summer.”                             | Sometimes the start of a question does not contain the question word. In this case, we scan the rest of the sentence to find the question word. |

# Datasets

## BoolQ

In the paper "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions", researchers gathered 16,000 boolean questions. Passages were collected from Wikipedia and questions were natural questions. Natural questions are written by people who did not know the answer to the question, as opposed to those who are prompted to write a particular kind of question. In other words, they are realistic questions people would actually ask in a real world context. 

![img](/site_files/evan-post-imgs/BoolQ_Dataset.png)

## MNLI

The Multi-Genre Natural Language Inference (MNLI) corpus contains 433K sentence pairs annotated with textual entailment information. The MNLI corpus covers a wide range of genres.

![img](/site_files/evan-post-imgs/mnli_dataset.png)

## SQuAD

The SQuAD dataset is a general question answering dataset. It contains some boolean questions, which we will steal to create neutral samples.

## Combined Dataset

**Entail:**

- boolq entail converted to phrase
- mnli entail

**Neutral:**

- squad questions (converted to phrase) + boolq contexts
- mnli neutrals

**Contradict:**

- boolq contradict converted to phrase
- mnli contradict

Our artificial neutral samples are SQUAD boolean questions paired with the most similar BoolQ context to that question. The assumption is that the squad question and boolq context will be on the same topic, except the boolq context will not contain the direct answer to the question. Similarity was calculated using the TF-DF algorithm in Elasticsearch. We input artificial neutral samples so that each label is mixed with contexts from both boolq and mnli datasets, combatting any arbitrary differences in the data distribution (e.g. mnli contexts are generally shorter than boolq contexts).

## Model Training

### roberta.large.mnli
RoBERTa is a transformers language model with 110M parameters that builds on [BERT](https://arxiv.org/abs/1706.03762). See the paper [here](https://arxiv.org/abs/1907.11692). RoBERTa was trained on the MultiNLI corpus and achieved a [90.2% accuracy on the mismatched development set for MNLI](https://paperswithcode.com/sota/natural-language-inference-on-multinli). We will refer to this model as roberta.large.mnli.
For training, we fine-tune roberta.large.mnli on this new dataset for 10 epochs with a batch size of 24.

# Results 

## Preliminary tests

We did two preliminary tests to justify some of our design choices for the fine-tuning process

**1) Converting to phrases improves accuracy for MNLI.**

We tested the accuracy for roberta.large.mnli on the boolq dev set. Like the boolq training set, the BoolQ dev set only contains True/False labels, so any Neutral predictions by any model is wrong. Here, we treated the entail prediction as the True label and contradict prediction as the False label. **With question words,** the accuracy was **54%**, and **with phrases** the accuracy was **60%.** 

**2) RoBERTa achieves state of the art accuracy on boolean question answering.** 

We trained roberta.mnli on top of BoolQ without neutral samples and obtained an accuracy of 85.84% on the boolq dev set. This beats the benchmark set in the paper of 80.4%, which fine-tunes a BERT-MNLI model on BoolQ.

## Results and Discussion

Our final results beats the benchmark set in the BoolQ paper by 3% with the added ability to predict neutral, while remembering general entailment information as evidenced by our models performance on the mnli mismatched dev set. Removing the artificial neutrals in the training process drops the accuracy for neutral predictions. Accuracy on boolq dev set was 83.82% and accuracy on mnli mismatched was 0.84.45%, a 6 percent drop from the baseline of 90% from roberta-mnli. 

_A comparison of flat accuracies_

| Model                               | BoolQ Dev Set (Question Word)                | BoolQ Dev Set (Phrase)| MNLI Mis-matched Dev Set |
| ----------------------------------- | -------------------------------------------- | --------------------- | -------------------------| 
| bert.mnli + boolq (paper benchmark) | 80.4                                         | -                     | -                        | 
| roberta.large.mnli                  | 27.21                                        | 60.3                  | 90.2                     | 
| roberta.large.mnli + boolq(Qword)   | 85.8                                         | -                     | -                        |  
| roberta.large.mnli + custom dataset | -                                            | 83.8                  | 84.5                     |


_Some example predictions for the context: "Given a boolean question and a passage, our task is to infer whether the answer to that question is Yes/No based on the information present in the passage. Boolean question answering has useful applications in smart search systems. Using NLMatics, a user can ask a boolean question and obtain the answer within a large document."_
| Original Question                                            | Phrase                                                   | Model Prediction                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| do we want to build a model that can answer yes/no in all situations, regardless whether a passage is present? | we do want to build a model that can answer yes/no in all situations, regardless whether a passage is present. | False  |
| Is boolean question answering currently used at NLMatics? | boolean question answering is current used at nlmatics.       | True |
| does NLMatics have users?                     | nlmatics does have users                             | True | 
| was the BoolQ dataset developed by Google?                      | the boolq dataset was developed by google.                             | Neutral |
| is NLMatics a text analytics startup?                      | nlmatics is a text analytics startup                             | Neutral |

## Concluding note
Boolean question answering is an exciting task that aims to instill common-sense reasoning skills into machines. This article aims to show that the core task behind boolean question answering is textual entailment. Future work can involve creating a generalized entailment model that works for both questions and claims. 

----------------
