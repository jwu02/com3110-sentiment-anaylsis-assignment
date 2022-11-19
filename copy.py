# list of prior probabilities p(s_i) for all sentiment classes
p_priors = [training_labels.count(c)/len(training_labels) for c in range(number_classes)]
likelihoods = {} # dict mapping sentiment class to word to its likelihood p(T|s_i)

for i in range(len(training_data)):
    sample = training_data[i]
    sentiment_label = training_labels[i]
    if sentiment_label not in WORD_OCCURRENCES:
        WORD_OCCURRENCES[sentiment_label] = {} # dict mapping word to occurrences
    
    for w in sample:
        if w not in WORD_OCCURRENCES[sentiment_label]:
            WORD_OCCURRENCES[sentiment_label][w] = 1
        else:
            WORD_OCCURRENCES[sentiment_label][w] += 1