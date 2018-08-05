import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """START_OF_SENTENCE The mathematician ran .
START_OF_SENTENCE The mathematician ran to the store .
START_OF_SENTENCE The physicist ran to the store .
START_OF_SENTENCE The philosopher thought about it .
START_OF_SENTENCE The mathematician solved the open problem .
""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {word_to_ix[word]: word for word in word_to_ix}

# build the model
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epoch_num = 1000
for epoch in range(epoch_num):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))


        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)

print('The hyper-parameters are: learning rate: ' + str(learning_rate) + ', epoch number: ' +str(epoch_num) +'.')

# Sanity check
check_sentence = """START_OF_SENTENCE The mathematician ran to the store .""".split()

check_trigrams = [([check_sentence[i], check_sentence[i + 1]], check_sentence[i + 2])
            for i in range(len(check_sentence) - 2)]

n = 0

for word, label in check_trigrams:
    word_idxs = torch.tensor([word_to_ix[w] for w in word], dtype=torch.long)
    out = model(word_idxs)
    _, predict_label = torch.max(out, 1)
    predict_word = ix_to_word[int(predict_label[0])]
    
    if predict_word == label:
        n += 1
        print("real word is '" + label + "', predict word is '" + predict_word + "', we get the right answer." )
    else:
        print("real word is '" + label + "', predict word is '" + predict_word + "', we get the wrong answer." )

    
accuracy = n / (len(check_trigrams))  
print('The accuracy of the model for the sanity check is ' + str(accuracy) + '.')  

# Test    
sentence_test = "START_OF_SENTENCE The ______ solved the open problem.".split()
location = int(sentence_test.index("______"))
trigrams_test = ([check_sentence[location-2], check_sentence[location- 1]], check_sentence[location])
word_idxs_test = torch.tensor([word_to_ix[w] for w in trigrams_test[0]], dtype=torch.long)
out_test = model(word_idxs_test)
log_physicist = out_test[0][word_to_ix['physicist']]
log_philosopher = out_test[0][word_to_ix['philosopher']]
if log_philosopher >= log_physicist:
    print("'physicist' is more likely to fill the gap. ")
else:
    print("'philosopher' is more likely to fill the gap. ")

embeds = nn.Embedding(len(word_to_ix), EMBEDDING_DIM)
mathematician_tensor = torch.tensor([word_to_ix["mathematician"]], dtype=torch.long)
mathematician_embed = embeds(mathematician_tensor)
physicist_tensor = torch.tensor([word_to_ix["physicist"]], dtype=torch.long)
physicist_embed = embeds(physicist_tensor)
philosopher_tensor = torch.tensor([word_to_ix["philosopher"]], dtype=torch.long)
philosopher_embed = embeds(philosopher_tensor)

cos = nn.CosineSimilarity()
cos_mat_phy = cos(mathematician_embed, physicist_embed)
cos_mat_phi = cos(mathematician_embed, philosopher_embed)

if cos_mat_phy > cos_mat_phi:
    print('the embeddings for "physicist" and "mathematician" are closer together according to the cosine similarity.')
else:
    print('the embeddings for "philosopher" and "mathematician" are closer together according to the cosine similarity.')