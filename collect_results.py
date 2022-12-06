import os

TRAINING_FILE = "moviereviews/train.tsv"
DEV_FILE = "moviereviews/dev.tsv"
TEST_FILE = "moviereviews/test.tsv"

NUMBER_CLASSES_CHOICES = [5, 3]
FEATURES_CHOICES = ['all_words', ' features']

for NUMBER_CLASSES in NUMBER_CLASSES_CHOICES:
    for FEATURES in FEATURES_CHOICES:
        os.system(f"""python NB_sentiment_analyser.py {TRAINING_FILE} {DEV_FILE} {TEST_FILE} -classes {NUMBER_CLASSES} -features {FEATURES} -output_files -confusion_matrix""")
        print("=============================================")