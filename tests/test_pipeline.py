import pandas as pd
from src.topic_model import LDATopicModel

def test_fit():
    df=pd.read_csv('data/demo_news.csv')
    model=LDATopicModel(n_topics=5,n_restarts=1).fit(df.text)
    assert model.theta.shape==(100,5)
    assert all(len(x)==5 for x in model.top_words(5))
    assert all(abs(sum(row)-1)<1e-5 for row in model.theta)
