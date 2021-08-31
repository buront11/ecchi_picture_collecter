import tweepy
import os

import urllib

CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_SECRET = os.getenv('ACCESS_SECRET')

def auth_twitter():
    # apiにアクセス
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, proxy='http://proxy.nagaokaut.ac.jp:8080')
    return api

def twitter_search(s):
    api = auth_twitter()

    tweets = tweepy.Cursor(api.search, q=s,
                            tweet_mode='extended', # 省略されたツイートを全て取得
                            include_entities=True,
                            result_type='mixed').items(limit=10)

    for tweet in tweets:
        # print(tweet.full_text)
        try:
            media = tweet.entities['media']
        except:
            continue
        print(media)

if __name__=='__main__':
    twitter_search('アズールレーン')