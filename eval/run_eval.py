import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.topic_model import LDATopicModel
from src.metrics import category_metrics

def main():
    base=Path(__file__).resolve().parents[1]
    df=pd.read_csv(base/'data/demo_news.csv')
    model=LDATopicModel(n_topics=5,n_restarts=2).fit(df.text)
    output=base/'outputs'; output.mkdir(exist_ok=True)
    topics=pd.DataFrame([{'topic':i+1,'word':word,'probability':prob} for i,items in enumerate(model.top_words(5)) for word,prob in items])
    topics.to_csv(output/'top_words.csv',index=False)
    mixtures=df.copy()
    for i in range(model.n_topics): mixtures[f'topic_{i+1}']=model.theta[:,i]
    mixtures['dominant_topic']=model.theta.argmax(axis=1)+1
    mixtures.to_csv(output/'document_topics.csv',index=False)
    summary={'documents':len(df),'vocabulary':len(model.vocab),'tokens':int(model.X.sum()),'perplexity':model.perplexity(),'selected_seed':model.seed,**category_metrics(df.category,mixtures.dominant_topic)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
