# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
from naive_bayes import NaiveBayes
from utilities import Utilities

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "mea20jw" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
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
    utilities = Utilities(USER_ID, number_classes, features, confusion_matrix)

    # training dataset
    training_ids, training_data, training_labels = utilities.load_and_preprocess_data(training)
    nb_model = NaiveBayes()
    nb_model.fit(training_data, training_labels)

    # utilities.print_document_frequencies()

    # dev dataset
    dev_ids, dev_data, dev_labels = utilities.load_and_preprocess_data(dev)
    predicted_dev_labels = nb_model.predict(dev_data)

    # test dataset
    test_ids, test_data, test_labels = utilities.load_and_preprocess_data(test)
    predicted_test_labels = nb_model.predict(test_data)

    if output_files:
        utilities.save_predictions(dev_ids, predicted_dev_labels, 'dev')
        utilities.save_predictions(test_ids, predicted_test_labels, 'test')

    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = utilities.evaluate_performance(predicted_dev_labels, dev_labels)
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()