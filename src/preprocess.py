import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
NEWS_STOP = set('said says say told mr mrs ms year years new news people also would could one two three last first time today yesterday according report reports'.split())
class Preprocessor:
    def __init__(self, remove_news_stop=True, lemmatize=True):
        self.stop=set(ENGLISH_STOP_WORDS) | (NEWS_STOP if remove_news_stop else set())
        self.lemmatize=lemmatize
        self.lemma=None
        if lemmatize:
            try:
                from nltk.stem import WordNetLemmatizer
                from nltk.corpus import wordnet
                wordnet.ensure_loaded()
                self.lemma=WordNetLemmatizer()
            except (ImportError, LookupError):
                pass
    def __call__(self,text):
        words=re.findall(r'[a-z]{3,}',str(text).lower())
        result=[]
        for word in words:
            if word in self.stop: continue
            if self.lemma: word=self.lemma.lemmatize(word)
            if word not in self.stop: result.append(word)
        return result
