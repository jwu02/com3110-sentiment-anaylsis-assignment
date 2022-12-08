class NaiveBayes():
    """
    Naive Bayes model for sentiment analysis
    """

    def __init__(self) -> None:
        self.number_classes = 0 # number of sentiment class labels

        # initialise model parameters
        # list of prior probabilities p(s_i) for all sentiment classes
        self.priors = []
        # dictionary mapping sentiment class to word, to its likelihood p(T|s_i)
        self.likelihoods = {}
        self.num_distinct_features = 0 # vocabulary size of training data


    def fit(self, training_data: list, training_labels: list) -> None:
        """
        Fit the model to the supplied training data to obtain the model parameters
        """
        self.number_classes = len(set(training_labels))

        # list of prior probabilities p(s_i) for all sentiment classes
        self.priors = [training_labels.count(c)/len(training_labels) for c in range(self.number_classes)]

        # dict mapping sentiment class to word to its likelihood p(T|s_i)
        self.likelihoods = {}

        # obtain word occurrences for each sentiment class first...
        for i in range(len(training_data)):
            sample = training_data[i]
            sentiment_class = training_labels[i]
            if sentiment_class not in self.likelihoods:
                self.likelihoods[sentiment_class] = {} # dict mapping word to occurrences
            
            for w in sample:
                if w not in self.likelihoods[sentiment_class]:
                    self.likelihoods[sentiment_class][w] = 1
                else:
                    self.likelihoods[sentiment_class][w] += 1

        # count number of distinct words/tokens in entire training corpus
        vocabulary = set()
        for sentiment_class in self.likelihoods:
            vocabulary.update(self.likelihoods[sentiment_class].keys())
        self.num_distinct_features = len(vocabulary)

        # ... in order to calculate the likelihood
        for sentiment_class in self.likelihoods:
            # counts all words/features of for a class (include duplicates)
            class_occurrences_sum = sum(self.likelihoods[sentiment_class].values())
            for w in self.likelihoods[sentiment_class]:
                # turn word counts into likelihood, with Laplace smoothing applied
                self.likelihoods[sentiment_class][w] = (self.likelihoods[sentiment_class][w] + 1) / \
                                                    (class_occurrences_sum + self.num_distinct_features)
    

    def predict(self, data: list) -> list:
        """
        Given a list of preprocessed data, returns a list of prediction labels
        the model has assigned to each data sample
        """
        predicted_labels = []

        for sample in data:
            sample_posteriors = []

            for sentiment_class in range(self.number_classes):
                class_occurrences_sum = sum(self.likelihoods[sentiment_class].values())
                sample_likelihoods = []
                for w in sample:
                    if w in self.likelihoods[sentiment_class]:
                        sample_likelihoods.append(self.likelihoods[sentiment_class][w])
                    else: # assign non-zero likelihood to words that have not appeared in training data
                        sample_likelihoods.append(1 / (class_occurrences_sum * self.num_distinct_features))
                
                # calculate product of all word likelihoods
                likelihood = 1
                for x in sample_likelihoods:
                    likelihood *= x

                # calculate and record posterior
                sample_posteriors.append(self.priors[sentiment_class] * likelihood)

            # assign class label as index of the maxmimum posterior
            predicted_labels.append(sample_posteriors.index(max(sample_posteriors)))

        return predicted_labels