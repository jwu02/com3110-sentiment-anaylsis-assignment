## Libraries used
- `nltk` a library designed for NLP usage, we only make use of its `stopwords` list
    - if running program for the first time, uncomment `nltk.download('stopwords')` to download the list of stopwords

## Command
- `python NB_sentiment_analyser.py moviereviews/train.tsv moviereviews/dev.tsv moviereviews/test.tsv -classes 5 -features features -confusion_matrix -output_files`
    - `-classes` choice from `[5, 3]`
    - `-features` choice from `[all_words, features]`

## Notes (ignore section)
### README contents
- contains all details about implementation needed to run code
- include libraries used
- detail installation of non-standard libraries

### Ideas
- preprocessing
    - lowercasing
    - any other techniques?
- feature selection
    - select certain emotion words based off another list, instead of using stoplist - sentiment lexicon
        - sentiwordnet
        - mpqa https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
    - intro to speech and language processing c4.4 optimizing
        - https://web.stanford.edu/~jurafsky/slp3/4.pdf p8
        - binarization, negation
    - https://arxiv.org/pdf/1911.00288.pdf
        - count difference (CD)
        - document frequency - manually remove high occurring and uninformative words?