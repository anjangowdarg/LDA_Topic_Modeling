import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from .preprocess import Preprocessor
class LDATopicModel:
    def __init__(self,n_topics=5,n_restarts=2,min_df=2,max_df=0.9,random_state=42,remove_news_stop=True,lemmatize=True):
        self.n_topics=n_topics; self.n_restarts=n_restarts; self.min_df=min_df; self.max_df=max_df; self.random_state=random_state
        self.preprocessor=Preprocessor(remove_news_stop,lemmatize)
    def fit(self,docs):
        self.docs=list(docs)
        if len(self.docs)<2: raise ValueError('At least two documents required')
        self.vectorizer=CountVectorizer(tokenizer=self.preprocessor,token_pattern=None,lowercase=False,min_df=self.min_df,max_df=self.max_df)
        self.X=self.vectorizer.fit_transform(self.docs)
        if self.X.shape[1]<2: raise ValueError('Too few vocabulary terms. Lower min_df or raise max_df.')
        best=None
        for i in range(self.n_restarts):
            model=LatentDirichletAllocation(n_components=self.n_topics,learning_method='batch',max_iter=100,random_state=self.random_state+i)
            theta=model.fit_transform(self.X); score=model.score(self.X)
            if best is None or score>best[0]: best=(score,model,theta,self.random_state+i)
        self.score,self.lda,self.theta,self.seed=best
        self.vocab=self.vectorizer.get_feature_names_out()
        self.topic_word=self.lda.components_/self.lda.components_.sum(axis=1,keepdims=True)
        return self
    def top_words(self,n=5):
        return [[(str(self.vocab[i]),float(row[i])) for i in np.argsort(row)[::-1][:n]] for row in self.topic_word]
    def document_topics(self): return self.theta
    def perplexity(self): return float(self.lda.perplexity(self.X))
