import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
text = open('Treasure_Island.txt','r')
pos_word = open('positive-words.txt', 'r')
neg_word = open('negative-words.txt', 'r')
posSentences = re.split(r'\n', pos_word.read())
negSentences = re.split(r'\n', neg_word.read())
 
posFeatures = []
negFeatures = []
    #http://stackoverf

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer( analyzer = 'word',tokenizer = tokenize, lowercase = True,stop_words = 'english',
    max_features = 85)
score = SentimentIntensityAnalyzer().polarity_scores('text')
print(score)