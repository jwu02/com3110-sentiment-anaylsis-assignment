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
    parser.add_argument("-classes", type=int, default=5, choices=[5, 3])
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

    def load_and_preprocess_data(filename: str) -> tuple:
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
                sentiment_label = line[2]
                
                data.append(processed_sentence)
                labels.append(sentiment_label)

        return (data, labels)


    def preprocess_sentence(sentence: str) -> list:
        sentence = sentence.split("\s")

        return sentence


    def get_model() -> tuple: # training
        # load and preprocess data
        training_data, training_labels = load_and_preprocess_data(training)
        # dev_data, dev_labels = load_and_preprocess_data(dev)
        # different preprocess process for test data file since no sentiment labels
        # test_data = load_and_preprocess_data(test)

        # list of prior probabilities p(s_i) for all sentiment classes
        p_priors = [training_labels.count(c)/len(training_labels) for c in range(number_classes)]
        likelihoods = {} # dict mapping sentiment class to word to its likelihood p(T|s_i)

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

        for sentiment_class in likelihoods:
            num_features = sum(likelihoods[sentiment_class].values())
            for w in likelihoods[sentiment_class]:
                # turn word counts into relative frequency/likelihood
                likelihoods[sentiment_class][w] /= num_features
            
        return (p_priors, likelihoods)


    model = get_model()

    


    
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