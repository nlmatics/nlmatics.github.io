import spacy

nlp = spacy.load("en_core_web_sm-2.3.1")

boolq_words = {
    "is",
    "isn't",
    "was",
    "were",
    "do",
    "does",
    "will",
    "can",
    "are",
    "has",
    "have",
    "had",
    "did",
    "should",
    "would",
    "could",
    "need",
}

other_qwords = {"who", "what", "where", "when", "how", "which", "why", "whom", "whose"}


#the goal is to place it in between nonsubj and a verb
#will there be a sequel 

def swap(doc, subject_index, qword, placement="After"):
    '''
        Function: swaps the question word in front of our after a desired character

        inputs:
            doc - spacy doc type
            subject_index - int, the index of the token that you want to put the question word in front / after
            qword - string, boolean question word
            placement - default "After" to put question word after desired token, anything else is before
        output:
            string of re-formatted question
    '''
    output_text = ""
    for i, token in enumerate(doc):
        if i > 0:
            if i == subject_index:
                if placement == "After":
                    output_text += (token.text + " " + qword + " ")
                else:
                    output_text += (qword + " " + token.text + " ")
            else:
                if token.tag_ == "POS" or token.dep_ == "punct" or token.text == "'s":

                    output_text = output_text.strip()

                output_text += token.text + " "

    return output_text

def reformat(question, first_word):
    '''
        input: question text, question word
        output: phrase text
    '''
    doc = nlp(question)
    
    #first priority is finding the noun subject of the question or "_qword__ there ___?"
    for i, token in enumerate(doc):
        if token.tag_ == "EX" and i == 1:
            question = swap(doc, i, first_word)
            return question.lower()
        if token.dep_ == "nsubj":
            if i < len(doc)-1 and doc[i+1].text == "and":
                while i < len(doc):
                    if doc[i].pos_ == "NOUN":
                        question = swap(doc, i, first_word)
                        return question
                    i+=1
            if i < len(doc)-1 and ((doc[i+1].pos_ != "PART" and doc[i+1].pos_ != "PUNCT") or doc[i+1].dep_ == "neg"):
                question = swap(doc, i, first_word)
                return question
        
    #then we try to place the question word in between a NOUN and verb/adj
    for i, token in enumerate(doc):
        if (token.pos_ == "VERB" or token.pos_ == "DET" or token.pos_ =="AUX") and i > 0 and (doc[i-1].pos_ == "NOUN" or doc[i-1].pos_ == "PRON" or doc[i-1].pos_ == "PROPN" or token.pos_ == "PRON"):
            question = swap(doc, i-1, first_word)

            return question
                                     
        if token.pos_ == "ADJ" and i > 0 and (doc[i-1].pos_ == "NOUN" or doc[i-1].pos_ == "PRON" or doc[i-1].pos_ == "PROPN"):
            question = swap(doc, i-1, first_word)

            return question
    #if we still can't find the above pattern, just place it after the first continuous block of nouns
    for i, token in enumerate(doc):
        found_noun = (token.pos_ == "NOUN" or token.pos_ == "PROPN")
        if found_noun and i < len(doc)-1 and doc[i+1].pos_ != "NOUN" and doc[i+1].pos_ != "PROPN" and doc[i+1].text != "and" and doc[i+1].pos_ != "PART" and doc[i+1].pos_ != "PUNCT":
            question = swap(doc, i, first_word)
            return question
    
    return question

def q2p(question):
    output = ""
    question = question.replace('?', '')
    question = question.lower()
    tokens = question.split(' ')
    first_word = tokens[0]
    
    if first_word not in boolq_words:
        #then either the boolean question is later on in the phrase OR this 
        if first_word in other_qwords:
            return question + '.'
        else:
            for i, word in enumerate(tokens):
                if word in boolq_words:
                    return " ".join(tokens[:i]).strip() + " " + reformat(" ".join(tokens[i:]), word).strip() + '.'
            return question + '.'

    return reformat(question, first_word).strip() + '.'