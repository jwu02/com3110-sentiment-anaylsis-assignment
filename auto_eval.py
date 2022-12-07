"""
Short script to automate evaluation process of model under 
different conditions: feature choice and number of classes
"""
import os

TRAINING_FILE = "moviereviews/train.tsv"
DEV_FILE = "moviereviews/dev.tsv"
TEST_FILE = "moviereviews/test.tsv"

NUMBER_CLASSES_CHOICES = [5, 3]
FEATURES_CHOICES = ['all_words', ' features']
OUTPUT_FILES = True
CONFUSION_MATRIX = True

for NUMBER_CLASSES in NUMBER_CLASSES_CHOICES:
    for FEATURES in FEATURES_CHOICES:
        command = f"python NB_sentiment_analyser.py {TRAINING_FILE} {DEV_FILE} {TEST_FILE} -classes {NUMBER_CLASSES} -features {FEATURES}"
        if OUTPUT_FILES:
            command += " -output_files"
        if CONFUSION_MATRIX:
            command += " -confusion_matrix"

        os.system(command)
        print("=============================================")