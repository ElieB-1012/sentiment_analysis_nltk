import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
import numpy as np
import re
import string
import random
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')



def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove the stock market tickers
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove the old styles retweet text 'RT'
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove the hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove the # symbol
    tweet = re.sub(r'#', '', tweet)

    # Tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []

    # removing stopwords and punctuation
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming
            tweet_clean.append(stem_word)

    return tweet_clean


def count_tweets(tweets, ys):
    ys_list = np.squeeze(ys).tolist()
    freqs = {}

    for y, tweet in zip(ys_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]
    return n


# splitting the data for training and testing
train_pos = all_positive_tweets[:4999]
test_pos = all_positive_tweets[4999:]

train_neg = all_negative_tweets[:4999]
test_neg = all_negative_tweets[4999:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# numpy array for the labels in the training set
train_y = np.append(np.ones((len(train_pos))), np.zeros((len(train_neg))))
test_y = np.append(np.ones((len(test_neg))), np.zeros((len(test_neg))))
print(test_y)

# Build a frequency dictionary
freqs = count_tweets(train_x, train_y)


def train_naive_bayes(freqs, train_y):
    logliklihood = {}

    # calculate V, number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    ## Calculate N_pos, N_neg, V_pos, V_neg
    # N_pos : total number of positive words
    # N_neg : total number of negative words
    # V_pos : total number of unique positive words
    # V_neg : total number of unique negative words

    N_pos = N_neg = V_pos = V_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            V_pos += 1
            N_pos += freqs[pair]
        else:
            V_neg += 1
            N_neg += freqs[pair]

    # Number of Documents (tweets)
    D = len(train_y)

    # D_pos, number of positive documnets
    D_pos = len(list(filter(lambda x: x > 0, train_y)))

    # D_pos, number of negative documnets
    D_neg = len(list(filter(lambda x: x <= 0, train_y)))

    # calculate the logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freqs_pos = lookup(freqs, word, 1)
        freqs_neg = lookup(freqs, word, 0)

        # calculate the probability of each word being positive and negative
        p_w_pos = (freqs_pos + 1) / (N_pos + V)
        p_w_neg = (freqs_neg + 1) / (N_neg + V)

        logliklihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, logliklihood


logprior, loglikelihood = train_naive_bayes(freqs, train_y)


def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    p = 0
    p += logprior

    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]

    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    y_hats = []
    goodResult=0
    for i in range(len(test_y)):
        if naive_bayes_predict(test_x[i], logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        if y_hat_i == test_y[i]:
            goodResult += 1
        y_hats.append(y_hat_i)
    error = 1 - goodResult/len(test_y)
    accuracy = 1 - error
    print(goodResult, len(test_x))

    return accuracy

categorized_tweets = ([[t, 1] for t in twitter_samples.strings("positive_tweets.json")] +
                      [[t, 0] for t in twitter_samples.strings("negative_tweets.json")])

def foldCrossValidation(categorized_tweets):
    accuracy = 0
    for i in range(10):
        random.shuffle(categorized_tweets)
        trainx = []
        temp = categorized_tweets[:9000]
        for i in range(len(temp)):
            trainx.append(temp[i][0])
        testx = []
        temp = categorized_tweets[9000:]
        for i in range(len(temp)):
            testx.append(temp[i][0])

        trainy = []
        temp = categorized_tweets[:9000]
        for i in range(len(temp)):
            trainy.append(temp[i][1])
        testy = []
        temp = categorized_tweets[9000:]
        for i in range(len(temp)):
            testy.append(temp[i][1])

        freq = count_tweets(trainx, trainy)
        prior, likelihood = train_naive_bayes(freq, trainy)
        accuracy += test_naive_bayes(testx, testy, prior, likelihood)
    print('foldcross accuracy: {}', accuracy/10)

foldCrossValidation(categorized_tweets)
"""
tweet = input("Enter a text: ")
p = naive_bayes_predict(tweet, logprior, loglikelihood)
if p > 0:
    print('Good sentiment.', p)
elif p < 0:
    print('Bad sentiment.', p)
else:
    print('Neutral sentiment', p)


"""