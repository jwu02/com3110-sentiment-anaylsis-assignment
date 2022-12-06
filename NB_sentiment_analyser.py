# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import csv
from naive_bayes import NaiveBayes
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "mea20jw" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    # parser.add_argument("-training", default="moviereviews/train.tsv")
    # parser.add_argument("-dev", default="moviereviews/dev.tsv")
    # parser.add_argument("-test", default="moviereviews/test.tsv")
    parser.add_argument("-classes", type=int)
    # parser.add_argument("-classes", type=int, default=5, choices=[5, 3])
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
        Load data from given filename and returns a tuple of a list of 
        preprocessed data samples, and a list of sentiment class labels
        """
        sentence_ids = [] # sentence ids
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
                
                sentence_ids.append(line[0])
                data.append(processed_sentence)
                labels.append(sentiment_label)
        
        return (sentence_ids, data, labels)


    def preprocess_sentence(sentence: str) -> list:
        sentence = sentence.split(" ")
        sentence = [w.lower() for w in sentence]

        MY_STOP_LIST = [',', '.', '--', '\'s', '...', '!']

        if features == 'features':
            stopwords_list = set(stopwords.words('english'))
            sentence = [w for w in sentence if w not in stopwords_list]
            sentence = [w for w in sentence if w not in MY_STOP_LIST]

        return sentence


    def evaluate_performance(predicted_labels, actual_labels) -> float:
        """
        Evaluate performance of system by the macro F1 score metric
        """
        # initialise confusion matrix with zeroes
        confusion_matrix_counts = [[0 for i in range(number_classes)] for j in range(number_classes)]

        correct = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == actual_labels[i]:
                correct += 1
            
            confusion_matrix_counts[actual_labels[i]][predicted_labels[i]] += 1
        
        print(f"Score: {correct} out of {len(predicted_labels)} correct. (REMOVE PRINT LATER)")

        # print confusion matrix if chosen to print it out
        if confusion_matrix:
            print("Confusion matrix:")
            for row in confusion_matrix_counts:
                print(row)

        macro_f1_scores = []
        for class_label in range(number_classes):
            # true positive
            tp = confusion_matrix_counts[class_label][class_label]

            other_classes = list(range(number_classes))
            other_classes.remove(class_label)
            
            # false positives
            fps = [confusion_matrix_counts[other_class_label][class_label] for other_class_label in other_classes]
            # false negatives
            fns = [confusion_matrix_counts[class_label][other_class_label] for other_class_label in other_classes]

            class_macro_f1_score = 2*tp / (2*tp+sum(fps)+sum(fns))
            macro_f1_scores.append(class_macro_f1_score)

        # return mean of macro-F1 scores across all classes
        return sum(macro_f1_scores) / len(macro_f1_scores)


    def save_results(sentence_ids: list, predicted_labels: list, dataset_name: str) -> None:
        """
        Save sentence ids and their corresponding predictions to tsv file
        """
        lines_to_write = []
        lines_to_write.append("SentenceID\tSentiment\n")
        for i in range(len(sentence_ids)):
            lines_to_write.append(f"{sentence_ids[i]}\t{predicted_labels[i]}\n")

        output_filename = f'{dataset_name}_predictions_{number_classes}classes_{USER_ID}.tsv'
        with open(output_filename, 'w') as f:
            f.writelines(lines_to_write)


    training_ids, training_data, training_labels = load_and_preprocess_data(training)
    nb_model = NaiveBayes()
    nb_model.fit(training_data, training_labels)

    dev_ids, dev_data, dev_labels = load_and_preprocess_data(dev)
    predicted_dev_labels = nb_model.predict(dev_data)
    save_results(dev_ids, predicted_dev_labels, 'dev')

    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = evaluate_performance(predicted_dev_labels, dev_labels)
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()