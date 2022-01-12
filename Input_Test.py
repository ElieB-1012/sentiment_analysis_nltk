# http://www.nltk.org/_modules/nltk/sentiment/util.html
# http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost

from sklearn.svm import LinearSVC
import random
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk import word_tokenize

stop = list(set(stopwords.words('English')))

categorized_tweets = ([(t, "pos") for t in twitter_samples.strings("positive_tweets.json")] +
                      [(t, "neg") for t in twitter_samples.strings("negative_tweets.json")])

smilies = [':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
           ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
           '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
           'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
           '<3', ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
           ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
           ':c', ':{', '>:\\', ';(', '(', ')', 'via']

'''categorized_tweets_tokens = []
for tweet in categorized_tweets:
    text = tweet[0]
    for smiley in smilies:
        text = re.sub(re.escape(smiley), '', text)
    categorized_tweets_tokens.append((word_tokenize(text), tweet[1]))
'''
import re


def processTweet(tweet):
    # process the tweets

    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', '', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')
    # remove numbers
    tweet = re.sub('[0-9]', '', tweet)
    return tweet


# end
for i in range(0, len(categorized_tweets)):
    categorized_tweets_remove = processTweet(categorized_tweets[i][0])
categorized_tweets = ([[t, "pos"] for t in twitter_samples.strings("positive_tweets.json")] +
                      [[t, "neg"] for t in twitter_samples.strings("negative_tweets.json")])
clean_tweets = []
for i in range(0, len(categorized_tweets)):
    clean_tweets.append([processTweet(categorized_tweets[i][0]), categorized_tweets[i][1]])

vocabulary = [w.lower() for i in range(0, len(clean_tweets)) for w in word_tokenize(clean_tweets[i][0]) if
              w.lower() not in stop and w.lower() not in smilies]
vocabulary = list(set(vocabulary))
vocabulary.sort()
random.shuffle(clean_tweets)
train = clean_tweets
tweet = input("Enter a tweet: ")
print('Predicting.. please wait.. ')
test = [[processTweet(tweet), 'pos']]

def get_unigram_features(data, vocab):
    fet_vec_all = []
    for tup in data:
        single_feat_vec = []
        sent = tup[0].lower()  # lowercasing the dataset
        for v in vocab:
            if sent.__contains__(v):
                single_feat_vec.append(1)
            else:
                single_feat_vec.append(0)
        fet_vec_all.append(single_feat_vec)
    return fet_vec_all


from nltk.corpus import sentiwordnet as swn


def get_senti_wordnet_features(data):
    fet_vec_all = []
    for tup in data:
        sent = tup[0].lower()
        words = sent.split()
        pos_score = 0
        neg_score = 0
        for w in words:
            senti_synsets = swn.senti_synsets(w.lower())
            for senti_synset in senti_synsets:
                p = senti_synset.pos_score()
                n = senti_synset.neg_score()
                pos_score += p
                neg_score += n
                break  # take only the first synset (Most frequent sense)
        fet_vec_all.append([float(pos_score), float(neg_score)])
    return fet_vec_all


def merge_features(featureList1, featureList2):
    # For merging two features
    if featureList1 == []:
        return featureList2
    merged = []
    for i in range(len(featureList1)):
        m = featureList1[i] + featureList2[i]
        merged.append(m)
    return merged


def get_lables(data):
    labels = []
    for tup in data:
        if tup[1].lower() == "neg":
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def calculate_precision(prediction, actual):
    prediction = list(prediction)
    correct_labels = [predictions[i] for i in range(len(predictions)) if actual[i] == predictions[i]]
    precision = float(len(correct_labels)) / float(len(prediction))
    return precision


training_unigram_features = get_unigram_features(train, vocabulary)  # vocabulary extracted in the beginning
training_swn_features = get_senti_wordnet_features(train)

training_features = merge_features(training_unigram_features, training_swn_features)

training_labels = get_lables(train)

test_unigram_features = get_unigram_features(test, vocabulary)
test_swn_features = get_senti_wordnet_features(test)
test_features = merge_features(test_unigram_features, test_swn_features)

test_gold_labels = get_lables(test)
# SVM Classifier
# Refer to : http://scikit-learn.org/stable/modules/svm.html

svm_classifier = LinearSVC(penalty='l2', C=0.01).fit(training_features, training_labels)
print("Prediction of linear SVM classifier is:")
predictions = svm_classifier.predict(test_features)
print("Prediction Test data\t" + str(predictions))

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
NBClassfier = clf.fit(training_features, training_labels)
print("Prediction of  Naive Bayes Regression is:")
predictions = NBClassfier.predict(test_features)
print("Prediction Test data\t" + str(predictions))
