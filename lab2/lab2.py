import nltk
from collections import Counter
from string import punctuation
import sys
import re

with open(sys.argv[1], 'r', encoding='utf-8') as text:
    text = text.read()
    text = text.lower()    #lowercase the texts
    text = re.sub(r'[{}]+'.format(punctuation),'',text)    #remove punctuation
    
text_words = text.split()
unigram_counts = Counter(text_words)   

text_sents=text.splitlines()    # bulid the list of sentences
sents = []
for i in range(len(text_sents)):
    sents.append(text_sents[i].split())

bigrams = []
for sent in sents:
    bigrams.extend(nltk.bigrams(sent, pad_left=True, pad_right=True))
bigram_counts = Counter(bigrams)

def bigram_LM(sentence_x, smoothing=0.0):
    unique_words = len(unigram_counts.keys()) + 2 
    x_bigrams = nltk.bigrams(sentence_x, pad_left=True, pad_right=True)
    prob_x = 1.0
    for bg in x_bigrams:
        if bg[0] == None:
            prob_bg = (bigram_counts[bg]+smoothing)/(len(sents)+smoothing*unique_words)
        else:
            prob_bg = (bigram_counts[bg]+smoothing)/(unigram_counts[bg[0]]+smoothing*unique_words)
        prob_x = prob_x *prob_bg
    return prob_x

with open(sys.argv[2], 'r', encoding='utf-8') as que:
    que = que.read()
    que = que.lower()   
questions = que.splitlines()

print("The answers using the unigram model are：\n")
for question in questions:
    words = question.split()
    answers = words[-1].split('/')
    p1 = unigram_counts[answers[0]]/len(text_words)
    p2 = unigram_counts[answers[1]]/len(text_words)
    print(answers[0] + ': ' + str(p1))
    print(answers[1] + ': ' + str(p2))
    if p1 > p2:
        print("answer: " + answers[0] + "\n")
    elif p2 > p1:
        print("answer: " + answers[1] + "\n")
    else:
        print("answer: " + answers[0] + " and " + answers[1] + "have the same probability\n")

print("The answers using the bigram model are：\n")
for question in questions:
    words = question.split()
    answers = words[-1].split('/')
    sentence1 = question.replace('____', answers[0])
    sentence2 = question.replace('____', answers[1])
    sentence1 = sentence1.split(':')
    sentence1 = sentence1[0]
    sentence1 = re.sub(r'[{}]+'.format(punctuation),'',sentence1)
    sentence1 = sentence1.split()
    p1 = bigram_LM(sentence1, smoothing=0.0)
    sentence2 = sentence2.split(':')
    sentence2 = sentence2[0]
    sentence2 = re.sub(r'[{}]+'.format(punctuation),'',sentence2)
    sentence2 = sentence2.split()
    p2 = bigram_LM(sentence2, smoothing=0.0)
    print(answers[0] + ': ' + str(p1))
    print(answers[1] + ': ' + str(p2))
    if p1 > p2:
        print("answer: " + answers[0] + "\n")
    elif p2 > p1:
        print("answer: " + answers[1] + "\n")
    else:
        print("answer: " + answers[0] + " and " + answers[1] + "have the same probability\n")

print("The answers using the bigram with add-1 smoothing (Laplace) model are：\n")
for question in questions:
    words = question.split()
    answers = words[-1].split('/')
    sentence1 = question.replace('____', answers[0])
    sentence2 = question.replace('____', answers[1])
    sentence1 = sentence1.split(':')
    sentence1 = sentence1[0]
    sentence1 = re.sub(r'[{}]+'.format(punctuation),'',sentence1)
    sentence1 = sentence1.split()
    p1 = bigram_LM(sentence1, smoothing=1.0)
    sentence2 = sentence2.split(':')
    sentence2 = sentence2[0]
    sentence2 = re.sub(r'[{}]+'.format(punctuation),'',sentence2)
    sentence2 = sentence2.split()
    p2 = bigram_LM(sentence2, smoothing=1.0)
    print(answers[0] + ': ' + str(p1))
    print(answers[1] + ': ' + str(p2))
    if p1 > p2:
        print("answer: " + answers[0] + "\n")
    elif p2 > p1:
        print("answer: " + answers[1] + "\n")
    else:
        print("answer: " + answers[0] + " and " + answers[1] + "have the same probability\n")
