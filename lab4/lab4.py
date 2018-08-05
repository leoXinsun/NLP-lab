import os
import random
import re
from collections import Counter
import time
import matplotlib.pyplot as plt
import sys


t0 = time.clock()
path = os.getcwd()

# create a list of neg_file_name
path_neg = path + "/" + sys.argv[1] + "/txt_sentoken/neg"   
files_neg= os.listdir(path_neg)
data_neg = []  
for file_neg in files_neg:
    item_neg = (file_neg, -1)
    data_neg.append(item_neg)

# create a list of pos_file_name
path_pos = path + "/" + sys.argv[1] + "/txt_sentoken/pos"   
files_pos= os.listdir(path_pos)
data_pos = []  
for file_pos in files_pos:
    item_pos = (file_pos, 1)
    data_pos.append(item_pos)

# create training dataset and testing dataset
train = int(len(data_neg) * 0.8)     
data_neg_train = data_neg[:train]
data_neg_test = data_neg[train:]
data_pos_train = data_pos[:train]
data_pos_test = data_pos[train:]

data_train = data_neg_train + data_pos_train
data_test = data_neg_test + data_pos_test

w = {}
accuracy_list = []

for i in range(0,15):
    w_list = []
    random.shuffle(data_train)
    # train the weight model
    for data in data_train:
        file_name = data[0]
        y = data[1]
        if y == -1:
            text = open(path + "/" + sys.argv[1] + "/txt_sentoken/neg/" + file_name).read()
        else:
            text = open(path + "/" + sys.argv[1] + "/txt_sentoken/pos/" + file_name).read()
        bag_of_words = Counter(re.sub("[^\w']"," ",text).split()[:])
        score = 0.0
        for word, counts in bag_of_words.items():
            if word not in w:
                w[word] = 0
            score += counts * w[word]
        if score >= 0:
            score = 1
        else:
            score = -1
        if score != y:
            for word, counts in bag_of_words.items():
                w[word] = w[word] + y * counts
        else:
            w = w
        w_list.append(w)

    # take the average of all the weight
    if i == 0:
        w_average = {}
        for key in w.keys():
            sum = 0
            for w_each in w_list:
                if key not in w_each:
                    value = 0
                else:
                    value = w_each[key]
                sum = sum + value
            value = sum / (len(w_list))
            w_average[key] = value

    else:
        for key in w.keys():
            sum = w_average[key] * (i * (len(w_list)))
            for w_each in w_list:
                value = w_each[key]
                sum = sum + value
            value = sum / ((i + 1) * (len(w_list)))
            w_average[key] = value

    # test the weight model
    correct = 0

    for data in data_test:
        file_name = data[0]
        y = data[1]
        if y == -1:
            text = open("C:/Users/canon/Desktop/review_polarity/txt_sentoken/neg/" + file_name).read()
        else:
            text = open("C:/Users/canon/Desktop/review_polarity/txt_sentoken/pos/" + file_name).read()
        bag_of_words = Counter(re.sub("[^\w']"," ",text).split()[:])
        score = 0.0
        for word, counts in bag_of_words.items():
            if word not in w_average:
                w_average[word] = 0
            score += counts * w_average[word]
        if score >= 0:
            score = 1
        else:
            score = -1
        if score == y:
            correct = correct + 1

    accuracy = correct / (len(data_test))
    accuracy_list.append(accuracy * 100)
    print("Through " + str(i+1) + " iteration, the accuracy is " + str(accuracy * 100) +"%.")
    print(str(i+1) + " iteration costs " + str(time.clock() - t0) + "s.\n")

# the top 10 features 
w_pos= sorted(w_average.items(), key=lambda d:d[1], reverse = True)
w_pos = w_pos[:10]
w_neg= sorted(w_average.items(), key=lambda d:d[1], reverse = False)
w_neg = w_neg[:10]
print("The 10 most positive items are:")
print(w_pos)
print("\nThe 10 most negative items are:")
print(w_neg)

#show the learning progress in a graph
x = list(range(1,16))
plt.plot(x,accuracy_list)
plt.title("the learning progress", fontsize=24)
plt.xlabel("iteration", fontsize=14)
plt.ylabel("accuracy(%)", fontsize=14)
plt.axis([0, 16, 50, 100]) 
plt.show()        



