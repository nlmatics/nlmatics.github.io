'''
Convert questions to phrases, in order to create a dataset of phrases and associated answers from a question-answer dataset.

(Code is written for use with SQuAD dataset -- it fixes certain mistakes in original SQuAD questions, and the main function parses a json file
written in SQuAD format. 
However, the code can easily be adapted to convert another dataset, 
or can be used to convert individual questions by passing them into the "choose method" function.)

*** NOTE: If using python 3.7 - in order to install the pattern library, 
run “brew install mysql”
“export PATH=$PATH:/usr/local/mysql/bin”
"pip install mysqlclient"
"pip install pattern"

Then, in the local installation of pattern, change line 609 of pattern->text->_init_.py from raise StopIteration to return

(if using python 3.6, pattern installation will work without modifying code)

*** to install spacy model used in this code, run "python -m spacy download en_core_web_lg"

'''

import json
from io import StringIO
import string
import spacy
from pattern.en import conjugate, PAST, PRESENT, tenses, lemma, lexeme, tag
nlp = spacy.load("en_core_web_lg")

question_words = {"who", "what", "where", "when", "how", "which", "why", "whom", "whose"}
count = 0
count_2 = 0

# fix pattern's conjugation of certain words into past tense
def change_tense_fixed(text, tense):
    if text == 'leave' and tense == PAST:
        return 'left'
    if text == 'quit' and tense == PAST:
        return 'quit'
    else: return conjugate(text, tense)

# identify if phrase is written in title case
def is_title_case(wordlist):
    is_title_case = [word.istitle() or word in ["of", "and", "for", "the", "to"] for word in wordlist]
    return all(is_title_case)

# convert sentence's verbs to past tense
def verbs_to_past_tense(textlist):
    newlist = []
    text = join_temp_text(textlist)
    # resolve discrepency between spacy and pattern's processing of words with "-" in middle - merge to one word
    if text.__contains__("-") and text[0][0] != "-" and text[0][-1]!="-":
        arr = text.split("-")
        arr[0]=arr[0]+(' -')
        newtxt = join_temp_text(arr)
        text = newtxt
    # use both pattern and spacy to catch verbs that one might miss 
    # (each seems more likely to misclassify nouns as verbs than other way around)
    doc = nlp(text) 
    tagged = tag(text)
    for idx, token in enumerate(doc):
        if (token.pos_ in ['VERB', "VBP", "VBZ"] or (len(doc)==len(tagged) and tagged[idx][1] in ["VB", "VBZ", "VBP"])) \
        and not token.text.istitle() and not (len(token.text) > 4 and token.text[-3:] == "ing") and \
        ((idx != 0 and token.nbor(-1).text not in ["to", "would","should","could"] and not token.nbor(-1).pos_ == "DET" and \
        (not ((idx !=1 and token.nbor(-1).nbor(-1).text in ["would", "should","could"]) or token.text in ["would","should","could"]))) or (idx == 0)):
            newlist.append(change_tense_fixed(token.text, PAST))
        else:
            newlist.append(token.text)
    return newlist

# conjugate verbs of phrase when removing word does
# e.g. why does the president _live_ in the white house -> the president _lives_ in the white house
def verbs_to_does(textlist):
    newlist = []
    text = join_temp_text(textlist)
    # change to incorporate pattern also?
    doc = nlp(text) 
    for idx, token in enumerate(doc):
        # print(token, token.pos_)
        if token.pos_ == 'VERB' and ((idx != 0 and token.nbor(-1).text not in ["to","are","is"] and not token.nbor(-1).pos_ == "DET") or (idx == 0)) and not token.text.istitle() and lemma(token.text) == token.text:
            if len(lexeme(token.text)) >= 2: newlist.append(lexeme(token.text)[1])
        elif token.text == "have": newlist.append("has")
        else:
            newlist.append(token.text)
    return newlist

# locate first verb in text
def findverb(text):
    word = ""
    verb_idx = -1
    
    # resolve discrepency between spacy and pattern's processing of words with "-" in middle - merge to one word
    if text.__contains__("-") and text[0][0] != "-" and text[0][-1]!="-":
        arr = text.split("-")
        arr[0]=arr[0]+(' -')
        newtxt = join_temp_text(arr)
        text = newtxt
    doc = nlp(text)
    tagged = tag(text)
    for ct, token in enumerate(doc):
        #print(tagged[ct])
        #print(token, token.pos_)
        if (token.pos_ in ["VERB", "VB","VB","VBG","VBN", "VBP", "VBZ"] or (len(doc)==len(tagged) and tagged[ct][1] in ["VB", "VBZ", "VBP", "VBG", "VBD"])) and not token.text.istitle() and ((ct != 0 and token.nbor(-1).text != "to" and not token.nbor(-1).pos_ == "DET") or ct == 0):
            verb_idx = ct
            word = token.text
            if ct != 0 and token.nbor(-1).pos_ in ["ADV", "ADJ"]:
                verb_idx -= 1
                word = token.nbor(-1).text
            break
    q_words = text.split()
    # accounts for disparity between splitting words at spaces and spacy splitting of words which makes punctuation individual tokens
    if verb_idx != -1 and (verb_idx > len(q_words) -1 or q_words[verb_idx] != word):
        if word in q_words:
            verb_idx = q_words.index(word)
        else:
            no_punct = text.translate(str.maketrans('', '', string.punctuation))
            no_punct_words = no_punct.split()
            if word in no_punct_words:
                verb_idx = no_punct_words.index(word)
    return verb_idx

# locate subject of text
def findsubj(text):
    s_idx = -1
    doc = nlp(text)
    for ct, token in enumerate(doc):
        #print(token, token.pos_)
        if token.dep_ == "nsubj":
            s_idx = ct
            break
    return s_idx

# clean (strip question mark, remove "the", append 's to words) and return array of words as string
def ret(q_words):
    newtext = " "
    q_words = list(filter(lambda x: x != "the", q_words))
    if "'s" in q_words:
        idx = q_words.index("'s")
        q_words[idx-1] = q_words[idx-1]+"'s"
        del q_words[idx]
    if "'nt" in q_words:
        idx = q_words.index("'nt")
        q_words[idx-1] = q_words[idx-1]+"'nt"
        del q_words[idx]
    if "'" in q_words:
        indices = [i for i, x in enumerate(q_words) if x == "'"]
        for index in indices:
            if index != 0:
                if q_words[index-1] == "'":
                    q_words[index-1] = "''"
        while "'" in q_words:
            q_words.remove("'")
    newtext = newtext.join(q_words).strip()
    if newtext.__contains__("?"): newtext = newtext.replace("?", "")
    if newtext.__contains__("  "): newtext = newtext.replace("  ", " ")

    return newtext

# join array of strings into string of text
def join_temp_text(arr):
    temptxt = " "
    return temptxt.join(arr)

# handle phrases with did or does 
def handle_did_does(q_words):
    # if sentence contains did and does, or did 2x or does 2x - only change phrases of words between first did/does and second
    def double_did_does(idx, endidx):
        if did_does == "did": firsthalf = verbs_to_past_tense(q_words[0:endidx])
        else: firsthalf = verbs_to_does(q_words[0:endidx])
        firsthalf.extend(q_words[endidx:])
        temptext = join_temp_text(firsthalf).split()
        return ret(temptext)
    did_does = ""
    both = False
    if "did" in q_words:
        did_does = "did"
    if "does" in q_words:
        if did_does == "did": 
            both = True
            # set did_does to first instance of word if both appear
            if q_words.index("does") < q_words.index("did"):
                did_does = "does"
        else: did_does = "does"
    idx = q_words.index(did_does)
    if q_words.count("did") == 2: # if two did's
        del q_words[idx]
        return double_did_does(idx, q_words.index("did"))
    elif q_words.count("does") == 2: # if two does
        del q_words[idx]
        return double_did_does(idx, q_words.index("does"))
    elif both: # if both
        if did_does == "did": endidx = q_words.index("does")
        else: endidx = q_words.index("did")
        del q_words[idx]
        return double_did_does(idx, endidx)
    else:
        if idx == len(q_words) -1: return ret(q_words) # if did/ does last word
        if q_words[idx+1] == 'not':
            # eg "which season did not have a group round in Hollywood?" -> "season that did not have a group round in Hollywood"
            if idx != 0: q_words.insert(idx, "that") 
            return ret(q_words)
        elif "not" in q_words:
            notidx = q_words.index("not")
            if idx < notidx:
                # should did be moved to before not
                # eg "Where did Beyonce not perform" -> "Beyonce did not perform"
                between = q_words[idx+1:notidx]
                doc = nlp(join_temp_text(between))
                not_verb = all(token.pos_ not in ["VERB", "VB","VB","VBG","VBN", "VBP", "VBZ"] for token in doc)
                if (len(between) == 1 and between[0] == 'he' or between[0] == 'she' or between[0] == 'they') or is_title_case(between) or not_verb:
                    q_words.insert(notidx, did_does)
                    q_words.remove(did_does)
                    return ret(q_words)
        # change conjugations of words after does/did, rejoin phrase
        if did_does == "does": lasthalf = verbs_to_does(q_words[idx+1:])
        if did_does == "did": lasthalf = verbs_to_past_tense(q_words[idx+1:])
        firsthalf = q_words[0:idx]
        firsthalf.extend(lasthalf)
        newtext = join_temp_text(firsthalf)
        return ret(newtext.split())

# handle questions beginning with which or what
def which_or_what(text): 
    # future improvement- if "is", "was" sentence doesn't have verb, insert that before and preserve "is"/"was" or put "is"/"was" after subj 
    # subj not correctly identified enough of the time by spacy to implmement this, and not sure how to distinguish between first case and second

    q_words = text.split()
    if q_words[0].lower() in ["which", "what"]: del q_words[0]
    if q_words[0] in ["is", "was", "do", "are", "has", 'had', "other", "can"]:
        del q_words[0]
        if q_words[0] == "been":
            del q_words[0]
        return ret(q_words)
    if "did" in q_words or "does" in q_words:
        return(handle_did_does(q_words))
    elif "was" in q_words:
        q_words.remove('was')
        return ret(q_words)
    elif "is" in q_words:
        q_words.remove('is')
        return ret(q_words)
    elif "are" in q_words:
        q_words.remove('are')
        return ret(q_words)
    elif "were" in q_words:
        q_words.remove('were')
        return ret(q_words)
    has_had = ""
    if "has" in q_words: has_had = "has" 
    if "had" in q_words: has_had = "had"
    if has_had != "":
        idx = q_words.index(has_had)
        if q_words[idx+1] == "been":
            del q_words[idx:idx+2]
            return ret(q_words)
        else: 
            # add that before has - eg "what word has many modern meanings" -> "word that has many modern meanings"
            idx = q_words.index(has_had)
            q_words.insert(idx, 'that')
            return(ret(q_words))
    else:
        return ret(q_words)

# handle questions beginning with when
def when(text):
    q_words = text.split()
    del q_words[0]
    # "when is it..." -> "it is..."
    if q_words[0] == "is":
        if q_words[1] == "it":
            q_words[0] = "it"
            q_words[1] = "is"
        else: del q_words[0]
    if (q_words[0] == "did" or q_words[0] == "does"):
        return(handle_did_does(q_words))
    elif q_words[0] in ["was", "were", "is","are"]:
        del q_words[0]
        if q_words[0] == "the":
            del q_words[0]
    # when being used as positional word, not as question word (ex "When America entered into the Cold War, what year was it?")
    elif text.__contains__(","):
        q_words.insert(0, "when")
        return other(text)
    return ret(q_words)

# handle questions beginning with who, whom, or whose
def who(text):
    q_words = text.split()
    if q_words[0].lower() in ["who", "whose","whom"]: del q_words[0]
    if "did" in q_words or "does" in q_words:
        return(handle_did_does(q_words))
    if q_words[0] in ["is", "was", "are", "else", "were"]:
        del q_words[0]
    return ret(q_words)

# handle questions beginning with has, have, does or did
def has_have_does_did(text):
    q_words = text.split()
    if q_words[0].lower() == "does":
        del q_words[0]
        return ret(verbs_to_does(q_words))
    if q_words[0].lower() == "did":
        del q_words[0]
        return ret(verbs_to_past_tense(q_words))
    else:
        if q_words[0].lower() in ["had", "have", "has"]: del q_words[0]
        return ret(q_words)

# handle questions beginning with where
# future improvement - something with would not, could not, should not, were not-- moving word to before not
def where(text):
    # move word in from of verb - ex. "where can strawberries grow" (word = can) -> "strawberries can grow"
    def is_was_are(word):
        q_words.remove(word)
        temptxt = join_temp_text(q_words)
        verbidx = findverb(temptxt)
        if verbidx != -1:
            q_words.insert(verbidx, word)
        elif q_words[-1] == "from":
            q_words.insert(-1, word)
    q_words = text.split()
    del q_words[0]
    if q_words[0] == 'did' or q_words[0] == 'does':
        return handle_did_does(q_words)
    if q_words[0] == "is":
        # "where is the setting/ location of..." -> "setting/ location of..."
        if len(q_words) >=3 and q_words[2] in ["setting", "location"]:
            del q_words[0:2]
            return ret(q_words)
        elif len(q_words) >=3 and q_words[2] == "there":
            if len(q_words)>=4 and q_words[3] == "a":
                del q_words[0:3]
                # "where is there a lot of" -> "place with a lot of"
                if q_words[0:1] == ["lot", "of"]:
                    del q_words[0:2]
                return ret(["place", "with"]+q_words)
        else: q_words.remove("is")
    elif q_words[0] == "are":
        if len(q_words)>=3 and q_words[2] == "there":
            del q_words[0:2]
            return ret(["there", "are"]+q_words)
        else: q_words.remove("are")
    elif q_words[0] == "was":
        if len(q_words) >=3 and q_words[2] == "there":
            if len(q_words)>=4 and q_words[3] == "a":
                del q_words[0:3]
            if q_words[0:1] == ["lot", "of"]:
                del q_words[0:2]
                return ret(["place", "with", "a", "lot", "of"]+q_words)
        else: is_was_are("was")
    elif q_words[0] == "were":
        q_words.remove("were")
    elif q_words[0] == "can":
        is_was_are("can")
    elif q_words[0] == "would":
        is_was_are("would")
    return ret(q_words)

# handle questions beginning with how many or how much (quantitative)
def how_many_much(text):
    def could_should_would(word):
        idx = q_words.index(word)
        # "that" before word + "be" - ex. "how many courses would be eaten" -> "number of courses that would be eaten"
        if q_words[idx+1] == "be":
            q_words.insert(idx, "that")
        # move word before verb - ex "how many chairs could the hall fit" -> "number of chairs the hall could fit"
        elif q_words[idx+1] != "not": 
            q_words.remove(word)
            verbidx = findverb(join_temp_text(q_words[idx+1:]))
            if verbidx != -1 and verbidx != 0:
                q_words.insert(verbidx, word)
            elif "hold" in q_words:
                q_words.insert(q_words.index("hold"),word)
            elif "worth" in q_words:
                q_words.insert(q_words.index("worth"),word)
    q_words = text.split()
    if q_words[0].lower() == "how": del q_words[0]
    if len(q_words) >= 1 and q_words[0] == "many":
        del q_words[0]
        if q_words[0] == "people":
            doc = nlp(join_temp_text(q_words))
            # how many people were there -> number of people who were there
            if doc[1].pos_ in ["AUX","VERB", "VB","VB","VBG","VBN", "VBP", "VBZ"]:
                q_words.insert(1, "who")
        # how many -> number of
        if q_words[0] not in ["will","should","could","were","are","did","does", "have","has","had","was","of"]:
            q_words = ["number","of"]+q_words
        else:
            # how many were etc. -> number that were...
            q_words = ['number']+q_words # number that?
    if len(q_words) >= 1 and q_words[0] == "much":
        del q_words[0]
        # how much -> amount of
        if q_words[0] not in ["will","should","could","was","is","did","does", "had", "has","have", "were","of"]:
            q_words = ["amount","of"]+q_words
        # ex. how much could she run -> amount that she could run
        elif q_words[0] not in ["more", "less"]:
            q_words = ['amount','that']+q_words
        else:
            q_words = ["amount"]+q_words
    if "do" in q_words and q_words[q_words.index("do")+1] != "not":
        q_words.remove("do")
    elif "were" in q_words and q_words[q_words.index("were")+1] != "not":
        q_words.remove("were")
    elif "did" in q_words and q_words[q_words.index("did")+1] != "not":
        q_words.remove("did")
        q_words = verbs_to_past_tense(q_words)
    elif "does" in q_words and q_words[q_words.index("does")+1] != "not":
        q_words.remove("does")
        q_words = verbs_to_does(q_words)
    elif "could" in q_words:
        could_should_would("could")
    elif "can" in q_words:
        could_should_would("can")
    elif "should" in q_words:
        could_should_would("should")
    elif "would" in q_words:
        could_should_would("would")
    elif "will" in q_words:
        could_should_would("will")
    elif "was" in q_words:
        q_words.remove("was")
    elif "had" in q_words:
        hadidx = q_words.index("had")
        if q_words[hadidx+1] == "been":
            del q_words[hadidx:hadidx+2]
        else:
            q_words.remove("had")
    elif "has" in q_words:
        hadidx = q_words.index("has")
        if q_words[hadidx+1] == "been":
            del q_words[hadidx:hadidx+2]
        else:
            q_words.remove("has")
    elif "is" in q_words: 
        q_words.remove("is")
    elif "are" in q_words: 
        areidx = q_words.index("are")
        if q_words[areidx+1] == "there":
            del q_words[areidx:areidx+2]
        else:
            q_words.remove("are")
    return ret(q_words)

# handle questions beginning with how
def how(text):
    def waswillis(word):
        # move word to front of verb
        # ex "how should they convert this phrase" -> "they should convert this phrase"
        if q_words[0] == 'it':
            del q_words[0]
        elif q_words[q_words.index(word)+1] == "it":
            del q_words[q_words.index(word)+1]
        verbidx = findverb(join_temp_text(q_words))
        if verbidx != -1 and verbidx != 0:
            q_words.insert(verbidx, word)
        q_words.remove(word)

    q_words = text.split()
    if q_words[0].lower() == "how": del q_words[0]
    # ex "how long was the movie" -> "length of the movie"
    if q_words[0] == 'long':
        del q_words[0]
        if q_words[0] in ["is", "was", "were","are"]:
            del q_words[0]
        if q_words[0] == "it":
            del q_words[0]
        q_words = ["length", "of"]+q_words
    # "how old" -> "age of"
    if q_words[0] == 'old':
        q_words[0] = "age"
        q_words[1] = "of"
    # "how large" -> "size of"
    if q_words[0] == 'large':
        del q_words[0]
        if q_words[0] in ["is", "was", "were","are"]:
            del q_words[0]
        q_words = ["size","of"]+q_words
    if "did" in q_words or "does" in q_words:
        return handle_did_does(q_words)
    elif q_words[0] == "has":
        waswillis("has")
    elif "do" in q_words:
        q_words.remove("do")
    elif "are" in q_words:
        q_words.remove("are")
    elif "were" in q_words:
        waswillis("were")
    elif "could" in q_words:
        waswillis("could")
    elif "can" in q_words:
        waswillis("can")
    elif "should" in q_words:
        waswillis("should")
    elif "would" in q_words:
        waswillis("would")
    elif "will" in q_words:
        waswillis("will")
    elif "was" in q_words:
        waswillis("was")
    elif "is" in q_words:
        waswillis("is")
    return ret(q_words)

# handle questions beginning with "why"
def why(text):
    q_words = text.split()
    # for SQuAD dataset - questions that have these as second word should begin with "what"
    if q_words[1] in {'causes','type','political','helps'}:
        q_words[0] = "What"
        return which_or_what(join_temp_text(q_words))
    if "did" in q_words or "does" in q_words:
        del q_words[0]
        return handle_did_does(q_words)
    if q_words[1] in ["exactly", "do"]: del q_words[1]
    why_words={'may', 'was', 'should', 'were', 'has', 'will', 'could', "can't", 'must', 'might', 'had', 'are', "didn't", 'is', 'did', "weren't", 'does', 'have', "wasn't", 'would', 'can', 'cant'}
    # move second word after subject - eg "why is the sky blue" -> "the sky is blue"
    if q_words[1] in why_words:
        temptxt = join_temp_text(q_words)
        s_idx = findsubj(temptxt)
        if s_idx != -1 and not s_idx == len(q_words)-1:
            # print("***")
            q_words.insert(s_idx+1, q_words[1])
        del q_words[1]
    if q_words[0].lower() == "why": del q_words[0]
    return ret(q_words)
    # add reason, reason that, move should could etc?

# handle questions not covered by categories above - 
# sentences with multiple clauses, sentences where questions words appear in middle/ at end of sentence, or other structure
def other(text):
    newtext = " "
    # if first word of clause to move to back isn't proper noun, make it lowercase
    def qwordinmiddle():
        doc = nlp(join_temp_text(firsthalf))
        if len(doc) > 0 and doc[0].pos_ != "PROPN": firsthalf[0] = firsthalf[0].lower()
    # split sentence at given word
    def splitatword(word):
        clause = text.split(word)
        lasthalf = clause[1].split()
        firsthalf = clause[0].split()
        return lasthalf, firsthalf
    q_words = text.split()
    if q_words[-1][:-1] in question_words or q_words[-1] in question_words or (q_words[-2] == 'how' and q_words[-1] in ["many", "much"]): 
            del q_words[-1] 
            if q_words[-2] in question_words:
                del q_words[-1:-3]
            elif q_words[-1] in question_words:
                del q_words[-1]
            return ret(q_words)
    elif q_words[0][:-2].lower() in question_words:
        q_words[0] = q_words[0][:-2]
        q_words.insert(0, "is")
        return choose_method(join_temp_text(q_words))
    # if no clear place to split sentence (eg no commas, semi-colons, or more than one comma), identify question word and split sentence there
    elif not (text.__contains__(",") or text.__contains__(";")) or text.count(",")>1:
        lasthalf = ""
        if "what" in q_words:
            if q_words[q_words.index("what")-1] in ["which", "where", "when"]:
                del q_words[q_words.index("what")-1]
            lasthalf, firsthalf = splitatword("what")
            lasthalf = which_or_what(join_temp_text(lasthalf)).split()
        elif "which" in q_words:
            lasthalf, firsthalf = splitatword("which")
            lasthalf = which_or_what(join_temp_text(lasthalf)).split()
        elif "who" in q_words or "whose" in q_words or "whom" in q_words:
            if "who" in q_words: lasthalf, firsthalf = splitatword("who")
            elif "whom" in q_words: lasthalf, firsthalf = splitatword("whom")
            elif "whose" in q_words: lasthalf, firsthalf = splitatword("who")
            temptxt = join_temp_text(lasthalf)
            lasthalf = who(join_temp_text(lasthalf)).split()
        elif "how" in q_words:
            if q_words[q_words.index("how")+1] in ["many", "much"]:
                lasthalf, firsthalf = splitatword(" how")
                lasthalf = how_many_much(join_temp_text(lasthalf)).split()
            else:
                lasthalf, firsthalf = splitatword(" how")
                lasthalf = how(join_temp_text(lasthalf)).split()
        elif "where" in q_words:
            lasthalf, firsthalf = splitatword("where")
            if len(lasthalf)>1:
                lasthalf = where(join_temp_text(lasthalf)).split()
        if lasthalf:
            qwordinmiddle()
            if not (len(firsthalf) == 1 and (lasthalf[-1] == firsthalf[0]) or lasthalf[-1]== "to" and firsthalf[0] =="with"): 
                lasthalf.extend(firsthalf)
            newtext = newtext.join(lasthalf)
            newarray = newtext.split()
            return ret(newarray)
        else: return ret(q_words)
    # split sentence at semi-colon or comma, treat second half as qeustion
    elif text.count(",") == 1 or text.__contains__(";"):
        if text.__contains__(";"):
            lasthalf, firsthalf = splitatword(";")
        else: lasthalf, firsthalf = splitatword(",")
        temptxt = join_temp_text(lasthalf)
        if lasthalf[0] in question_words:
            temptxt = choose_method(temptxt)
        elif lasthalf[0] == "does" or lasthalf[0] == "did":
            lasthalf = handle_did_does(lasthalf)
        elif any(x in lasthalf for x in question_words):
            temptxt = choose_method(temptxt)
        if temptxt != " ": 
            lasthalf = temptxt.split()
        qwordinmiddle()
        if not (len(firsthalf) == 1 and (lasthalf[-1] == firsthalf[0]) or lasthalf[-1]== "to" and firsthalf[0] =="with"): 
            lasthalf.extend(firsthalf)
        newtext = newtext.join(lasthalf)
        newarray = newtext.split()
        return ret(newarray)
    else:
        return ret(q_words)

# choose appropriate modification function for given question
def choose_method(question):
    text = ""
    q_tokens = question.split()
    #if q_tokens[0].lower() in before_what_words and q_tokens[1] in ['which', 'what']:
        #text = beforewhatwhich(question)
    if q_tokens[0].lower() in ['which','what']:
        text = which_or_what(question)
    elif q_tokens[0].lower() == 'when':
        text = when(question)
    elif q_tokens[0].lower() in ['who','whose','whom']:
        text = who(question)
    elif q_tokens[0].lower() in ["have","has","had","done","did"]:
        text = has_have_does_did(question)
    elif q_tokens[0].lower() == 'where':
        text = where(question)
    elif q_tokens[0].lower() == 'how':
        if q_tokens[1] in ["many", "much"]:
            text = how_many_much(question)
        else:
            text = how(question)
    elif q_tokens[0].lower() == 'why':
        text = why(question)
    #elif q_tokens[0].lower() in ["with", "to", "for", "from"] and q_tokens[1] == "whom":
        #text = with_whom(question)
    else:
        text = other(question)
    return text

# fix mistakes/typos in SQuAD datast questions
def fix_squad_irregularities(q_tokens, question):
    if 'dd' in q_tokens:
        print(question)
        q_tokens[q_tokens.index('dd')] = 'did'
        question = join_temp_text(q_tokens)
    if question.__contains__("what what"):
        print(question)
        q_tokens.remove("what")
        question = join_temp_text(q_tokens)
    if question.__contains__(" how how"):
        print(question)

        q_tokens.remove("how")
        question = join_temp_text(q_tokens)
    if question.__contains__(" was was "):
        print(question)

        q_tokens.remove("was")
        question = join_temp_text(q_tokens)
    if q_tokens[-1] == "?":
        q_tokens[-2] = q_tokens[-2]+'?'
        del q_tokens[-1]
        question = join_temp_text(q_tokens)
    if q_tokens[0] in ["WHy", "wy", "whi"]:
        print(question)
 
        q_tokens[0] = "Why"
        question = join_temp_text(q_tokens)
    if q_tokens[0][0] in ["`","'",'"']:
        q_tokens[0] == q_tokens[1:]
        question = join_temp_text(q_tokens)
    return q_tokens, question

# convert SQuAD dataset from phrases to questions, output json file in SQuAD format with phrases in place of questions
# to convert dev set replace filename with 'dev-v2.0.json', rename outfile
if __name__ == "__main__":

    with open('train-v2.0.json',) as f:
        squad = json.load(f)

        for i in squad["data"]:
            for ct, j in enumerate(i["paragraphs"]):
                # if creating sample dataset, you can uncomment the following line to take only the first quesiton on every article, for greater
                # variation in question topics
                #if ct > 1: break
                for k in j["qas"]:
                    # uncomment the following line to convert a subset of all SQuAD questions as sample, replace x with desired sample size
                    #if count >= 300: break
                    q_tokens = k["question"].split()
                    q_tokens, k["question"] = fix_squad_irregularities(q_tokens, k["question"])
                    
                    # while reformatting, print converted questions in format
                    # question count
                    # old question
                    # >> new phrase
                    print(k["question"])
                    k["question"] = choose_method(k["question"])
                    count += 1
                    assert(k["question"] != "None")
                    print(">>", k["question"])
                    print(count)

    with open('SQuAD_to_phrases_train.json', 'w') as outfile:
        json.dump(squad, outfile)
