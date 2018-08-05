from sklearn.metrics import f1_score
import sys

train = open(sys.argv[1]).read()
train = train.splitlines()

sentences = []

for sentence in train:
    sentence = sentence.split()
    lengh = int((len(sentence)) / 2 )
    sentence_1 = []
    for i in range(lengh):
        word_label = (sentence[i],sentence[i + lengh])
        sentence_1.append(word_label)
    sentences.append(sentence_1)

words = []
for sentence in sentences:
    for word_label in sentence:
        words.append(word_label[0]) 
words = set(words)

# First method
w_1 ={}    #build the model
for i in words:
    w_1[i] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
    

for i in range(10):    #train the model
    for sentence in sentences:
        for word_label in sentence:
            word = word_label[0]
            label = word_label[1]
            key_1 = w_1[word]
            if label != max(key_1,key=key_1.get):
                key_1[max(key_1,key=key_1.get)] -=  1
                key_1[label] += 1
                w_1[word] = key_1
            else:
                key_1[label] += 1

test = open(sys.argv[2]).read()
test = test.splitlines()

sentences_test = []

for sentence in test:
    sentence = sentence.split()
    lengh = int((len(sentence)) / 2 )
    sentence_1 = []
    for i in range(lengh):
        word_label = (sentence[i],sentence[i + lengh])
        sentence_1.append(word_label)
    sentences_test.append(sentence_1)

y_true = []
y_predicted = []    #predict the label
for sentence in sentences_test:
    for word_label in sentence:
        word = word_label[0]
        label = word_label[1]
        y_true.append(label)
        if word not in w_1:
            w_1[word] = {'O' : 0,
                    'PER' : 0,
                    'LOC' : 0,
                    'ORG' : 0,
                    'MISC' : 0,}
        key_2 = w_1[word]
        y_predicted.append(max(key_2,key=key_2.get))

f1_micro = f1_score(y_true, y_predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
print("First method: The micro-F1 score with the current word-current label feature is:")
print(f1_micro)

# the second method
w_2 ={}    #bulid model
for i in words:
    w_2[i] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
w_2['O'] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
w_2['PER'] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
w_2['LOC'] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
w_2['ORG'] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
w_2['MISC'] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
w_2['START'] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}

for i in range(10):   #train the modle
    for sentence in sentences:
        labels_predict = []
        for i in range(len(sentence)):
            word_label = sentence[i]
            word = word_label[0]
            label = word_label[1]
            if i == 0:
                label_previous = 'START'
            p = -100
            key_list = ['O', 'PER', 'LOC', 'ORG','MISC']
            for label_1 in key_list:
                p1 = w_2[word][label_1]
                p2 = w_2[label_previous][label_1]
                p12 = p1 + p2
                if p12 > p:
                    label_predict = label_1
                    p = p12
            labels_predict.append(label_predict)
            label_previous = label_predict
        for j in range(len(labels_predict)):
            word_current = sentence[j][0]
            w_2[word_current][labels_predict[j]] -= 1
            w_2[word_current][sentence[j][1]] += 1
            if j == 0:
                w_2['START'][labels_predict[j]] -= 1 
                w_2['START'][sentence[j][1]] += 1
            else:
                w_2[labels_predict[j-1]][labels_predict[j]] -= 1
                w_2[sentence[j-1][1]][sentence[j][1]] += 1

y_true = []
y_predicted = []
for sentence in sentences_test:   #predict the label
    for word_label in sentence:
        word = word_label[0]
        label = word_label[1]
        y_true.append(label)
        if word not in w_2:
            w_2[word] = {'O' : 0,
                    'PER' : 0,
                    'LOC' : 0,
                    'ORG' : 0,
                    'MISC' : 0,}
    for i in range(len(sentence)):
        word_label = sentence[i]
        word = word_label[0]
        label = word_label[1]
        key_2 = w_2[word]
        y_predicted.append(max(key_2,key=key_2.get))

f1_micro = f1_score(y_true, y_predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
print("Second method: The micro-F1 score with the current word-current label and previous label-current label feature is:")
print(f1_micro)   

# the third method
w_3 ={}    # bulid the model
words_previous = []

for word in words:
    word_previous = word + '_previous'
    words_previous.append(word_previous)

for i in words:
    w_3[i] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}
for i in words_previous:
    w_3[i] = {'O' : 0,
    'PER' : 0,
    'LOC' : 0,
    'ORG' : 0,
    'MISC' : 0,}

for i in range(10):    # train the model
    for sentence in sentences:
        labels_predict = []
        for j in range(len(sentence)):
            word_label = sentence[j]
            word = word_label[0]
            label = word_label[1]
            word_previous = sentence[j-1][0] + '_previous'
            word_previous_label = sentence[j-1][1]
            key_1 = w_3[word]
            key_2 = w_3[word_previous]
            if j != 0:
                p = -100
                key_list = ['O', 'PER', 'LOC', 'ORG','MISC']
                for label_1 in key_list:
                    p1 = w_3[word][label_1]
                    p2 = w_3[word_previous][label_1]
                    p12 = p1 + p2
                    if p12 > p:
                        label_predict = label_1
                        p = p12
            else:
                label_predict = max(key_1,key=key_1.get)
            labels_predict.append(label_predict)
        for k in range(len(labels_predict)):
            word_current = sentence[k][0]
            w_3[word_current][labels_predict[k]] -= 1
            w_3[word_current][sentence[k][1]] += 1
            if j != 0:
                word_previous_1 = sentence[k-1][0] + '_previous'
                w_3[word_previous_1][labels_predict[k]] -= 1
                w_3[word_previous_1][sentence[k][1]] += 1



y_true = []
y_predicted = []
for sentence in sentences_test:    #predict the label
    for word_label in sentence:
        word = word_label[0]
        label = word_label[1]
        y_true.append(label)
        if word not in w_3:
            w_3[word] = {'O' : 0,
                    'PER' : 0,
                    'LOC' : 0,
                    'ORG' : 0,
                    'MISC' : 0,}
        key_2 = w_3[word]
        y_predicted.append(max(key_2,key=key_2.get))

f1_micro = f1_score(y_true, y_predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
print("Third method: The micro-F1 score with the current word-current label and previous word-previous label feature is:")
print(f1_micro)

#get the top 10:
o_1 = {}
o_2 = {}
o_3 = {}
per_1 = {}
per_2 = {}
per_3 = {}
loc_1 = {}
loc_2 = {}
loc_3 = {}
org_1 = {}
org_2 = {}
org_3 = {}
misc_1 = {}
misc_2 = {}
misc_3 = {}

for word in words:
    o_1[word] = w_1[word]['O']
    o_2[word] = w_2[word]['O']
    o_3[word] = w_3[word]['O']
    per_1[word] = w_1[word]['PER']
    per_2[word] = w_2[word]['PER']
    per_3[word] = w_3[word]['PER']
    loc_1[word] = w_1[word]['LOC']
    loc_2[word] = w_2[word]['LOC']
    loc_3[word] = w_3[word]['LOC']
    org_1[word] = w_1[word]['ORG']
    org_2[word] = w_2[word]['ORG']
    org_3[word] = w_3[word]['ORG']
    misc_1[word] = w_1[word]['MISC']
    misc_2[word] = w_2[word]['MISC']
    misc_3[word] = w_3[word]['MISC']

o_1 = sorted(o_1.items(), key = lambda item:item[1], reverse = True)
o_2 = sorted(o_2.items(), key = lambda item:item[1], reverse = True)
o_3 = sorted(o_3.items(), key = lambda item:item[1], reverse = True)
per_1 = sorted(per_1.items(), key = lambda item:item[1], reverse = True)
per_2 = sorted(per_2.items(), key = lambda item:item[1], reverse = True)
per_3 = sorted(per_3.items(), key = lambda item:item[1], reverse = True)
loc_1 = sorted(loc_1.items(), key = lambda item:item[1], reverse = True)
loc_2 = sorted(loc_2.items(), key = lambda item:item[1], reverse = True)
loc_3 = sorted(loc_3.items(), key = lambda item:item[1], reverse = True)
org_1 = sorted(org_1.items(), key = lambda item:item[1], reverse = True)
org_2 = sorted(org_2.items(), key = lambda item:item[1], reverse = True)
org_3 = sorted(org_3.items(), key = lambda item:item[1], reverse = True)
misc_1 = sorted(misc_1.items(), key = lambda item:item[1], reverse = True)
misc_2 = sorted(misc_2.items(), key = lambda item:item[1], reverse = True)
misc_3 = sorted(misc_3.items(), key = lambda item:item[1], reverse = True)

top10_o_1 = []
top10_o_2 = []
top10_o_3 = []
top10_per_1 = []
top10_per_2 = []
top10_per_3 = []
top10_loc_1 = []
top10_loc_2 = []
top10_loc_3 = []
top10_org_1 = []
top10_org_2 = []
top10_org_3 = []
top10_misc_1 = []
top10_misc_2 = []
top10_misc_3 = []
for i in range(10):
    top10_o_1.append(o_1[i][0])
    top10_o_2.append(o_2[i][0])
    top10_o_3.append(o_3[i][0])
    top10_per_1.append(per_1[i][0])
    top10_per_2.append(per_2[i][0])
    top10_per_3.append(per_3[i][0])
    top10_loc_1.append(loc_1[i][0])
    top10_loc_2.append(loc_2[i][0])
    top10_loc_3.append(loc_3[i][0])
    top10_org_1.append(org_1[i][0])
    top10_org_2.append(org_2[i][0])
    top10_org_3.append(org_3[i][0])
    top10_misc_1.append(misc_1[i][0])
    top10_misc_2.append(misc_2[i][0])
    top10_misc_3.append(misc_3[i][0])

top10_1={}
top10_1['O']= top10_o_1
top10_1['PER'] = top10_per_1
top10_1['LOC'] = top10_loc_1
top10_1['ORG'] = top10_org_1
top10_1['MISC'] = top10_misc_1
top10_2={}
top10_2['O']= top10_o_2
top10_2['PER'] = top10_per_2
top10_2['LOC'] = top10_loc_2
top10_2['ORG'] = top10_org_2
top10_2['MISC'] = top10_misc_2
top10_3={}
top10_3['O']= top10_o_3
top10_3['PER'] = top10_per_3
top10_3['LOC'] = top10_loc_3
top10_3['ORG'] = top10_org_3
top10_3['MISC'] = top10_misc_3

print('The top 10 for the first method is:')
print(top10_1)
print('The top 10 for the second method is:')
print(top10_2)
print('The top 10 for the third method is:')
print(top10_3)