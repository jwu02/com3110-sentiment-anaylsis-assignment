# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import csv

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "mea20jw" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    # parser.add_argument("training")
    # parser.add_argument("dev")
    # parser.add_argument("test")
    parser.add_argument("-training", default="moviereviews/train.tsv")
    parser.add_argument("-dev", default="moviereviews/dev.tsv")
    parser.add_argument("-test", default="moviereviews/test.tsv")
    # parser.add_argument("-classes", type=int)
    parser.add_argument("-classes", type=int, default=3, choices=[5, 3])
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """

    MERGE_CLASSES_MAPPING = {
        0: 0, # negative            -> negative
        1: 0, # somewhat negative   -> negative
        2: 1, # neutral             -> neutral
        3: 2, # somewhat positive   -> positive
        4: 2, # posititve           -> positive
    }

    def load_and_preprocess_data(filename: str, test_data=False) -> tuple:
        """
        Load data from given filename, preprocess it and return
        a tuple of a list of sentences, and a list of sentiment labels
        """
        data = [] # sentences
        labels = [] # sentiments

        with open(filename) as f:
            read_data = csv.reader(f, delimiter="\t")
            next(read_data, None) # skip column headings and ignore return value
            for line in read_data:
                processed_sentence = preprocess_sentence(line[1])
                
                if number_classes == 5:
                    sentiment_label = int(line[2])
                else: # number_classes == 3
                    sentiment_label = MERGE_CLASSES_MAPPING[int(line[2])]
                
                data.append(processed_sentence)
                labels.append(sentiment_label)
        
        return (data, labels)


    def preprocess_sentence(sentence: str) -> list:
        sentence = sentence.split(" ")

        return sentence


    def get_model() -> dict: # training
        """
        Return a model fitted to the training data with following keys
        - priors: list of prior probabilities p(s_i) for all sentiment classes
        - likelihoods: dictionary mapping sentiment class to word to its likelihood p(T|s_i)
        - num_distinct_features: the vocabulary size of the training data
        """
        model = {}

        # load and preprocess data
        training_data, training_labels = load_and_preprocess_data(training)

        # list of prior probabilities p(s_i) for all sentiment classes
        priors = [training_labels.count(c)/len(training_labels) for c in range(number_classes)]
        model['priors'] = priors

        # dict mapping sentiment class to word to its likelihood p(T|s_i)
        likelihoods = {}

        # obtain word occurrences for each sentiment class first...
        for i in range(len(training_data)):
            sample = training_data[i]
            sentiment_class = training_labels[i]
            if sentiment_class not in likelihoods:
                likelihoods[sentiment_class] = {} # dict mapping word to occurrences
            
            for w in sample:
                if w not in likelihoods[sentiment_class]:
                    likelihoods[sentiment_class][w] = 1
                else:
                    likelihoods[sentiment_class][w] += 1

        # ... in order to calculate the likelihood
        vocabulary = set()
        for sentiment_class in likelihoods:
            vocabulary.update(likelihoods[sentiment_class].keys())
        num_distinct_features = len(vocabulary)
        model['num_distinct_features'] = num_distinct_features

        for sentiment_class in likelihoods:
            class_occurrences_sum = sum(likelihoods[sentiment_class].values())
            for w in likelihoods[sentiment_class]:
                # turn word counts into likelihood, with Laplace smoothing applied
                likelihoods[sentiment_class][w] = (likelihoods[sentiment_class][w] + 1) / \
                                                    (class_occurrences_sum + num_distinct_features)
        
        model['likelihoods'] = likelihoods

        return model


    def evaulate_dev(data: list) -> list:
        """
        Given a list of preprocessed data returns a list of labels
        the system has assigned to each data sample
        """
        # instantiate model represented by its parameters
        model = get_model()
        priors = model['priors']
        likelihoods = model['likelihoods']
        num_distinct_features = model['num_distinct_features']
        
        evaluation_labels = []

        for sample in data:
            sample_posteriors = []

            for sentiment_class in range(number_classes):
                class_occurrences_sum = sum(likelihoods[sentiment_class].values())
                sample_likelihoods = []
                for w in sample:
                    if w in likelihoods[sentiment_class]:
                        sample_likelihoods.append(likelihoods[sentiment_class][w])
                    else:
                        sample_likelihoods.append(1 / (class_occurrences_sum * num_distinct_features))
                
                # calculate product of all word likelihoods
                likelihood = 1
                for x in sample_likelihoods:
                    likelihood *= x

                # calculate and record posterior
                sample_posteriors.append(priors[sentiment_class] * likelihood)

            # assign class label as index of the maxmimum posterior
            evaluation_labels.append(sample_posteriors.index(max(sample_posteriors)))

        return evaluation_labels


    dev_data, dev_labels = load_and_preprocess_data(dev)
    evaluated_labels = evaulate_dev(dev_data)

    correct = 0
    for i in range(len(evaluated_labels)):
        if evaluated_labels[i] == dev_labels[i]:
            correct += 1
    
    print(f"Score: {correct} out of {len(evaluated_labels)} correct")

    
    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = 0
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()