# Training a Transformer QA Model on Phrase-Answer Pairs
By Batya Stein

## Introduction

Have you ever wondered what the most popular pizza topping in America is? To find out, you might Google “what is the most popular pizza topping in America?”. If you’re feeling lazy, maybe you’d leave out the question mark, or drop the question format altogether, and go with a simple “most popular pizza topping in America”. 

This probably seems like an excessive amount of thought to put into a Google search query, especially since all the above prompts return the exact same answer. (Pepperoni, for the curious.) With or without the question word and question mark, you can intuitively recognize that all these formulations are essentially asking for the same answer.   

However, imagine that you wanted to train a machine learning model to complete a similar exercise. If you fed your model background material on America’s pizza-eating habits, and then asked it some variation of the topping question, you’d like it to identify the answer “pepperoni” within the text. Models are usually trained for this type of question-answering task using reading comprehension datasets, like the **S**tanford **Q**uestion **A**nswering **D**ataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)). SQuAD’s examples consist of context paragraphs from Wikipedia, and questions which either have answers located within the context paragraphs, or are unanswerable. Each question is written as a full sentence, complete with a question mark at the end. Nearly all questions contain a question word (“who”, “what”, “where”, etc.). 

[](/site_files/batya-post-imgs/squad-data-pt.png)
_Example of SQuAD question and context paragraph_

Given that a model trained on SQuAD has only been exposed to prompts in the form of questions, the specific way a user words their query takes on new importance - would a model trained only on questions be able to process phrases and match them to answers with the same accuracy as it can questions? In other words, which characteristics are most necessary for a prompt to be accurately understood by a SQuAD-trained model? Though it may seem obvious to you and me that “what is the most popular topping?” is basically synonymous with “most popular topping”, it is not clear that the same would be apparent to a SQuAD-trained model.   

## Experiment Motivation

As an intern at NLMatics this summer, I ran a series of experiments using transformer based question-answering models and the SQuAD 2.0 dataset to explore how the wording of a dataset’s questions impacts a model’s accuracy. My goal was to see if: 
1. a model **trained** only on <ins>question</ins>-answer pairs would give accurate results when **evaluated** on <ins>phrase</ins>-answer pairs,
and 
2. if a model trained from scratch on phrase-answer pairs could reach the same accuracy scores as a question-trained model. 

I also trained models on different permutations of the phrase and question datasets - phrases with question marks, and questions without question marks - to try and isolate the characteristics that make a question answering model effective. Is it the presence of a question mark, or of a question word, or simply the keywords of a phrase?

The results of these experiments are relevant for any system where a user directly queries a text using a question-answering model. The designer of the system doesn’t necessarily have control over the questions users ask, or a way of knowing if they will choose to input questions (ex. “where is the building?”) or phrases (ex. “building location”). If a model could be trained for accuracy in phrase answering tasks without a significant decrease in accuracy on question answering tasks, users would be able to input a much broader range of questions, without the designer having to worry about precise wording. 

## Question to Phrase Conversion

In order to create a dataset of phrases directly comparable to a question dataset, I decided to reformat the SQuAD 2.0 dataset into phrases. ([Google Natural Questions](https://ai.google.com/research/NaturalQuestions), another large reading comprehension dataset, has some examples that are already written as phrases and some written as questions, but the variability makes it less useful for direct comparisons.)  My goal when converting the questions was to create phrases that sounded like something natural a user might type into a search engine, so that the training data would be similar to what would be seen in a real use case. To this end, I wrote a python script to process SQuAD that, besides deleting question words, changed tenses when necessary, moved verbs after subjects, and reordered the clauses of complex questions. I used the [spaCy](https://spacy.io/) NLP library for part-of-speech tagging and dependency parsing (ex. identifying the subject of a sentence), and the [pattern](https://github.com/clips/pattern#pattern) library for verb conjugation. The code had rules for handling questions based on the question words they began with, and a separate rule for handling questions with more complex structures.

Some example outputs from my phrase conversion code:

|Original Question|New Phrase|comment|
|---|---|---|
|“On what date did Henry Kissinger negotiate an Israeli troop withdrawal from the Sinai Peninsula?”|“date Henry Kissinger negotiated an Israeli troop withdrawal from Sinai Peninsula on”|To create a natural sounding phrase after deleting “did” from questions, I used the pattern library’s verb conjugation function to change remaining verbs to past tense
|“Atticus Finch's integrity has become a model for which job?”|“job Atticus Finch's integrity has become a model for”|For examples where question words appeared mid-sentence (instead of as the first word), I split questions at the question word and moved the first clause to the back|
“Where would present day Joara be?”|“present day Joara would be”|When a phrase wouldn’t make sense without an auxiliary verb like “would”, “could”, “can”, etc., I used spaCy to identify the question’s subject, and moved the auxiliary verb behind it (otherwise, I deleted unnecessary auxiliary verbs such as “are” or “is”)

## Model Training

To figure out which characteristics of a question were most important, I trained four Albert models on four different datasets. Albert is a light variation of the Bert transformer model with fewer parameters for an even quicker training time than Bert. (See the original paper on Bert [here](https://arxiv.org/abs/1706.03762), with a helpful explanation [here](https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/) and a general overview of transformer architecture [here](https://jalammar.github.io/illustrated-transformer/).) 

The four datasets I used were:
1. Normal SQuAD (2.0) dataset - a baseline for comparison
[](/site_files/batya-post-imgs/normal-q.png)
2. SQuAD converted to phrases - can the model match phrases to answers effectively since they are more open ended then questions?
[](/site_files/batya-post-imgs/phrse.png)
3. SQuAD questions without question marks - how important is question mark in getting the model to recognize a question?
[](/site_files/batya-post-imgs/q-no-mark.png)
4. SQuAD converted to phrases, with question marks appended - does the addition of a question mark compensate for the lack of a question structure?
[](/site_filename/batya-post-imgs/phrase-w-q.png)

The Albert XL model that I used came from huggingface’s [transformer library](https://github.com/huggingface/transformers), and for training I used huggingface’s [run_squad.py](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py) script. I used [wandb](https://www.wandb.com/), a web-based training visualization library, to monitor metrics throughout my trainings. 

## SQuAD Metrics

The official metrics for evaluating a SQuAD trained model are calculated in the [evaluation script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) linked on SQuAD’s website. The two main metrics are “exact” and “F1”. “Exact” returns the percentage of examples that the model predicted correct answers for out of all test examples. “[F1](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)” is an accuracy measure combining precision (number of true positives over total predicted positives) and recall (number of true positives over everything that should have been identified as positive). To calculate F1 scores for SQuAD, for each dev set example, precision is the number of overlapping words between the correct and predicted answers, divided by the number of words in the predicted answer. Recall is the number of overlapping words divided by the number of words in the correct answer.  

The metrics return F1 and exact scores over all dev set examples, as well as F1 and exact scores over the subset of dev set examples that have answers, (“HasAns”) and over the subset of examples that are impossible to answer (“NoAns”).

## Results and Discussion

To analyze the results of my 4 model trainings, I first compared the performances of each model evaluated on its own dev set (I used dev sets for evaluations since SQuAD’s test set is not publicly available). In other words, after training, how accurately would each model perform on data formatted the same way as its training data?

(For consistency, all bar graph results shown below are evaluated at the 7000th training step, towards the end of 2nd epoch. The exception is the F1 score chart, which spans from beginning to end of training.)


As expected, the **standard SQuAD-trained model** has high metrics across the board and high F1 scores throughout training, since its examples have the most context for the model to pick up on (question format and question marks). Interestingly, the **SQuAD without question marks model** does better than the standard model for the “HasAns” metrics, and notably worse for the “NoAns”- I am not sure exactly what to conclude from this (it may not be a conclusive trend, as the bar graph only represents one checkpoint from training). Overall though, the F1 scores of the SQuAD without question marks model were equal to, or higher than, the standard SQuAD F1’s, showing that the inclusion or exclusion of question marks seems relatively insignificant for achieving accuracy. 
The **phrase-trained model** metrics are equal to, or 1-3 points lower than the standard SQuAD metrics, and its F1 scores throughout training are lower overall but not to a super significant extent. This answers one of my main beginning questions, indicating that while there is some increased accuracy associated with the additional context of a question structure, <ins>a model trained on phrase-answer pairs does not show a significant decrease in accuracy.</ins>

For the second part of my analysis, I evaluated each of the models on the standard SQuAD dev set, and the standard SQuAD trained model on the phrase dev set. (All results evaluated at the 7000th step, as above). Through these evaluations I hoped to see how well a model trained on a specific modified dataset could perform on a differently formatted question. 


Notably, <ins>a standard SQuAD-trained model performs really poorly when evaluated on phrases</ins>, with an F1 drop of about 20, and 30-40 point drops in the “HasAns” categories. Practically, this proves the necessity of training models better equipped to deal with phrases. On a theoretical level, this indicates that the model comes to rely on question words/ structure for context when trained on questions, and therefore it can’t process questions accurately without them. (That’s not to say that question words are inherently necessary for achieving accuracy in training - see phrase-trained model metrics above - rather, once a model is trained on questions, it will become reliant on question characteristics).

On the other hand, as shown in the chart below, a model trained on phrases performs almost as accurately on standard questions as it does on its own dev set - it’s well adapted to inputs of both phrases and questions. It seems to do slightly better on the phrase evaluation because the model learns certain weights corresponding to the format of its training data, but since here the second dev set is introducing new information instead of taking away expected context, the extra input barely hurts.



One other interesting result came from the model trained on **questions without question marks**. When evaluated on **questions with question marks** (i.e. normal SQuAD) the metrics were slightly better than on its own dev set for the “HasAns” metrics, but slightly worse for the “NoAns”. I would guess this is because if the model isn’t exposed to question marks during training, but then is given a question with a question mark, it will be more likely to assume that something is an answer based on the extra input, which leads to increased accuracy if the question actually has an answer, but decreased accuracy when the question is impossible.


In conclusion, it seems like Bert transformer models begin to recognize and expect specific characteristics from their training data. This affects the results when a model is evaluated on data formatted differently from its original training set, but to a much less dramatic extent if information is being added rather than taken away. 

On the other hand, the initial dataset a model is trained on doesn’t affect accuracy as much as you might think it would - a model trained on phrases will be within a few points of a question-trained model’s accuracy, and adding or taking away question marks makes a mostly negligible difference.

Running these experiments was cool because I was able to get a better feel for how a model actually adapts to a dataset when training on it. I also think the results I found can be practically pretty helpful for anyone designing a question-answering system and wondering what types of input would be most effective.

## Further Developments 

There are a few ways to build on this project that I think could be interesting. You could train an Albert model twice, on normal SQuAD and then on SQuAD converted to phrases, to see if it achieves better results on both questions and phrases. However, training a model twice on the same data might lead to overfitting on SQuAD. An alternative would be to train a model using MAML, [Model Agnostic Meta Learning](https://towardsdatascience.com/model-agnostic-meta-learning-maml-8a245d9bc4ac), which allows a model to learn multiple different tasks, treating  questions and phrases as two separate tasks. It would also be interesting to train a model on the full [Google Natural Questions dataset](https://ai.google.com/research/NaturalQuestions), which is already a mixture of questions and phrases, and evaluate it just with a question dataset and just with a phrase dataset to see how the mixture and larger dataset size affects things.



Thank you to my supervisor, Sonia Joseph, who initially proposed this idea to me, helped along the way, and introduced me to MAML.

