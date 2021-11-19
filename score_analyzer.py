import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


def vader_score(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound']


def textblob_score(sentence):
    analysis = TextBlob(sentence).sentiment
    return analysis.polarity


def naive_bayes(sentence):
    blob_object = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
    analysis = blob_object.sentiment
    if analysis.classification == 'neg':
        return analysis.p_neg * -1
    else:
        return analysis.p_pos


for file in ['donaldtrump.csv', 'joebiden.csv']:
    with open(file, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tweet_count = {}
        vader_sentiment_score = {}
        text_blob_sentiment_score = {}
        bayes_sentiment_score = {}
        for row in csv_reader:
            if row:
                if row[0]:
                    if len(row[0]) > 10 and '2020' in row[0]:
                        date = ascii(row[0][0:10])
                        if date not in tweet_count:
                            vader_sentiment_score[date] = vader_score(row[2])
                            text_blob_sentiment_score[date] = textblob_score(row[2])
                            bayes_sentiment_score[date] = naive_bayes(row[2])
                            tweet_count[date] = 1
                        else:
                            vader_sentiment_score[date] += vader_score(row[2])
                            text_blob_sentiment_score[date] += textblob_score(row[2])
                            bayes_sentiment_score[date] += naive_bayes(row[2])
                            tweet_count[date] += 1

        df = pd.DataFrame()
        dates = tweet_count.keys()
        
        for key in vader_sentiment_score:
            vader_sentiment_score[key] = vader_sentiment_score[key] / tweet_count[key]
        for key in text_blob_sentiment_score:
            text_blob_sentiment_score[key] = text_blob_sentiment_score[key] / tweet_count[key]
        for key in bayes_sentiment_score:
            bayes_sentiment_score[key] = bayes_sentiment_score[key] / tweet_count[key]
        
        df['Date'] = dates
        df['Vader score'] = vader_sentiment_score.values()
        df['Text Blob score'] = text_blob_sentiment_score.values()
        df['Naive Bayes score'] = bayes_sentiment_score.values()
        filename = file[:-4]+'Score.csv'
        df.to_csv(filename, index=False)
