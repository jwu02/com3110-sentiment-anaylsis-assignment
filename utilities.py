import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class Utilities:
    """
    Class used to load the data, also includes a preprocessing stage and 
    a feature selection (if chosen) stage
    """
    
    def __init__(self, user_id: str, number_classes: str, features: str, confusion_matrix: bool) -> None:
        self.USER_ID = user_id
        self.number_classes = number_classes
        self.features = features
        self.confusion_matrix = confusion_matrix

        self.MERGE_CLASSES_MAPPING = {
            0: 0, # negative            -> negative
            1: 0, # somewhat negative   -> negative
            2: 1, # neutral             -> neutral
            3: 2, # somewhat positive   -> positive
            4: 2, # posititve           -> positive
        }

        self.stopwords_list = set()
        if features == 'features':
            self.PUNCTUATIONS = [',', '.', '...', '!', '?', '`', '``', '\'', '\'\'', '-', '--', ':', ';', '-lrb-', '-rrb-']

            self.stopwords_list = set(stopwords.words('english'))
            # # inspect stopwords list
            # for w in self.stopwords_list:
            #     print(w)

            # # words identified by examining top most frequent words
            # ADDITIONAL_STOPWORDS = []
            # self.stopwords_list.update(ADDITIONAL_STOPWORDS)
            
            # some words in stopwords list are needed for negation, if it's used
            # some tokens like `wasn't`` and `wasn`` are not included in the negation trigger words list
            # since the way the dataset tokenizes these words into `was` and `n't`
            self.NEGATION_TRIGGER_WORDS = ['n\'t', 'no', 'not', 'nor', 'neither', 'never', 'barely', 'hardly', 'scarcely', 'seldom', 'rarely']
            self.stopwords_list = self.stopwords_list.difference(self.NEGATION_TRIGGER_WORDS)
        
        # for manually examining most frequent terms
        self.current_filename = None
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.current_sentence_id = None


    def load_and_preprocess_data(self, filename: str) -> tuple:
        """
        Load data from given filename and returns a tuple of a list of 
        preprocessed data samples, and a list of sentiment class labels
        """
        self.current_filename = filename

        sentence_ids = [] # sentence ids
        data = [] # sentences
        labels = [] # sentiments

        with open(filename) as f:
            read_data = csv.reader(f, delimiter='\t')
            next(read_data, None) # skip column headings and ignore return value
            for line in read_data:
                sentence_ids.append(line[0])
                self.current_sentence_id = line[0]

                processed_sentence = self.preprocess_sentence(line[1])
                data.append(processed_sentence)
                
                # not given sentiment class labels for test data, so can't process the column
                if filename != 'moviereviews/test.tsv':
                    if self.number_classes == 5:
                        sentiment_label = int(line[2])
                    else: # number_classes == 3
                        sentiment_label = self.MERGE_CLASSES_MAPPING[int(line[2])]
                    labels.append(sentiment_label)
        
        return (sentence_ids, data, labels)


    def preprocess_sentence(self, sentence: str) -> list:
        """
        Basic preproessing step used for both all_words and features
        """
        processed_sentence = sentence.split(" ")
        processed_sentence = [w.lower() for w in processed_sentence]

        if self.features == 'features': # if chosen to use features
            processed_sentence = self.select_features(processed_sentence)
        
        # construct document frequency mapping of training data for inspection
        if self.current_filename == 'moviereviews/train.tsv':
            for w in processed_sentence:
                if w in self.term_frequencies:
                    self.term_frequencies[w] += 1
                else:
                    self.term_frequencies[w] = 1

        return processed_sentence


    def select_features(self, sentence: list) -> list:
        """
        Feature selection stage
        """
        selected_features = sentence
        selected_features = self.apply_stopwords(selected_features)
        selected_features = self.apply_negation(selected_features)
        # apply binarization after negation, so punctuation don't get removed
        # selected_features = self.apply_binarization(selected_features)

        return selected_features
    

    def print_document_frequencies(self):
        """
        Print top most occurring terms for inspection
        """
        for i in range(1000):
            top = max(self.term_frequencies, key=self.term_frequencies.get)
            print(f"{top}\t{self.term_frequencies[top]}")
            del self.term_frequencies[top]


    def apply_stopwords(self, sentence: list) -> list:
        """
        Uses a (slightly modified) stopwords list from NLTK library to 
        remove unwanted features
        """
        selected_features = []
        
        for w in sentence:
            # if w in stopwords_list or ('\'' in w and w != 'n\'t'):
            if w in self.stopwords_list:
                continue
            else:
                selected_features.append(w)
        
        return selected_features
    

    def apply_negation(self, sentence: list) -> list:
        negated_sentence = []

        negate = False
        for w in sentence:
            # start negating after encountering negation trigger word
            if not negate and w in self.NEGATION_TRIGGER_WORDS:
                negate = True
                negated_sentence.append(w)
                continue

            # stop negating at punctuations, also checks for double negation
            if negate and (w in self.PUNCTUATIONS or w in self.NEGATION_TRIGGER_WORDS):
                negate = False

            # if token not punctuation nor have apostrophe, append word 
            if not (w in self.PUNCTUATIONS or ('\'' in w and w != 'n\'t')):
                if negate: # negate by appending 'NOT_' to a word
                    negated_sentence.append(f'NOT_{w}')
                    continue
                else:
                    # an additional stopwords list but for punctuations, which are retained till 
                    # here so we can make use of negation
                    # '\'' = ['n\'t', '\'ve', '\'s', '\'re', '\'ll', '\'m', ...]
                    negated_sentence.append(w)
        
        # # check if negation modifies sentence of features accordingly
        # if self.current_sentence_id == '1872':
        #     print(negated_sentence)
        
        return negated_sentence


    def apply_binarization(self, sentence: list) -> list:
        """
        Binarization removes multiple occurrences of each word in each sentence, 
        which is basically taking the set of the list
        """
        return list(set(sentence))


    def evaluate_performance(self, predicted_labels: list, actual_labels: list) -> float:
        """
        Evaluate performance of system by the macro F1 score metric, 
        which is the average of F1 scores for all classes
        """
        # initialise confusion matrix with zeroes
        confusion_matrix_counts = [[0 for i in range(self.number_classes)] for j in range(self.number_classes)]

        correct = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == actual_labels[i]:
                correct += 1
            
            confusion_matrix_counts[actual_labels[i]][predicted_labels[i]] += 1
        
        # print(f"Score: {correct} out of {len(predicted_labels)} correct. (REMOVE PRINT LATER)")

        # print confusion matrix if chosen to print it out
        if self.confusion_matrix:
            print("Confusion matrix:")
            for row in confusion_matrix_counts:
                print(row)

        # computing the macro F1 score makes use of the confusion matrix
        f1_scores = []
        for class_label in range(self.number_classes):
            # true positive
            tp = confusion_matrix_counts[class_label][class_label]

            other_classes = list(range(self.number_classes))
            other_classes.remove(class_label)
            
            # false positives
            fps = [confusion_matrix_counts[other_class_label][class_label] for other_class_label in other_classes]
            # false negatives
            fns = [confusion_matrix_counts[class_label][other_class_label] for other_class_label in other_classes]

            f1_scores.append(2*tp / (2*tp+sum(fps)+sum(fns))) # append F1 score for each class

        # return macro F1 score which is the mean of F1 scores across all classes
        return sum(f1_scores) / len(f1_scores)


    def save_predictions(self, sentence_ids: list, predicted_labels: list, dataset_name: str) -> None:
        """
        Save sentence ids and their corresponding predicted sentiment class labels to tsv file
        """
        lines_to_write = []
        lines_to_write.append("SentenceID\tSentiment\n")
        for i in range(len(sentence_ids)):
            lines_to_write.append(f"{sentence_ids[i]}\t{predicted_labels[i]}\n")

        output_filename = f'{dataset_name}_predictions_{self.number_classes}classes_{self.USER_ID}.tsv'
        with open(output_filename, 'w') as f:
            f.writelines(lines_to_write)