import io
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from src.topic_model import LDATopicModel
from src.metrics import category_metrics
st.set_page_config(page_title='LDA Topic Explorer',layout='wide',page_icon='🧩')
st.title('🧩 LDA Topic Modeling Explorer')
st.caption('Discover topics, compare document mixtures, and export results. Different corpora, topic counts, and seeds produce different learned outputs.')
with st.sidebar:
    st.header('Corpus')
    source=st.radio('Source',['Bundled demo news','Paste documents','Upload CSV or TXT'])
    k=st.slider('Number of topics',2,10,5)
    top_n=st.slider('Top words per topic',3,15,5)
    restarts=st.slider('Random restarts',1,10,2)
    seed=st.number_input('Random seed',min_value=0,value=42)
    with st.expander('Advanced preprocessing'):
        min_df=st.number_input('Minimum document frequency',min_value=1,value=2)
        max_df=st.slider('Maximum document frequency',0.2,1.0,0.9,0.05)
        news_stop=st.checkbox('Remove news stop words',True)
        lemma=st.checkbox('WordNet lemmatization (if installed)',True)
if source=='Bundled demo news':
    df=pd.read_csv('data/demo_news.csv')
elif source=='Paste documents':
    raw=st.text_area('Paste documents separated by blank lines',height=230,placeholder='Document one...\n\nDocument two...')
    docs=[d.strip() for d in raw.split('\n\n') if d.strip()]
    df=pd.DataFrame({'doc_id':range(1,len(docs)+1),'text':docs})
else:
    file=st.file_uploader('Upload a CSV with a text column, or a TXT with blank-line-separated documents',type=['csv','txt'])
    if file is None: st.info('Upload a file to begin.'); st.stop()
    if file.name.lower().endswith('.csv'):
        df=pd.read_csv(file)
        column=st.selectbox('Text column',df.select_dtypes(include='object').columns.tolist())
        df=df.rename(columns={column:'text'})
    else:
        docs=[d.strip() for d in file.getvalue().decode('utf-8-sig').split('\n\n') if d.strip()]
        df=pd.DataFrame({'doc_id':range(1,len(docs)+1),'text':docs})
if 'text' not in df: st.error('No text column found'); st.stop()
df=df.dropna(subset=['text']).reset_index(drop=True)
a,b,c=st.columns(3)
a.metric('Documents',len(df)); b.metric('Words',sum(len(str(x).split()) for x in df.text)); c.metric('Topics requested',k)
if st.button('Discover Topics',type='primary'):
    try:
        model=LDATopicModel(k,restarts,int(min_df),max_df,int(seed),news_stop,lemma).fit(df.text)
        st.session_state['result']=(model,df.copy(),(source,k,restarts,seed,min_df,max_df,news_stop,lemma))
    except Exception as exc: st.error(f'Cannot fit model: {exc}')
if 'result' not in st.session_state: st.stop()
model,df,settings=st.session_state['result']
if settings!=(source,k,restarts,seed,min_df,max_df,news_stop,lemma): st.warning('Settings changed. Click Discover Topics to refit the model.')
st.success(f'Fitted {len(df)} documents | vocabulary {len(model.vocab)} | tokens {int(model.X.sum())} | selected seed {model.seed} | perplexity {model.perplexity():.1f}')
top=model.top_words(top_n)
summary=pd.DataFrame([{'Topic':f'Topic {i+1}','Top words':', '.join(w for w,p in words),'Probabilities':', '.join(f'{p:.3f}' for w,p in words),'Dominant documents':int((model.theta.argmax(axis=1)==i).sum())} for i,words in enumerate(top)])
st.subheader('1. Top words by topic'); st.dataframe(summary,use_container_width=True)
st.download_button('Download topic summary CSV',summary.to_csv(index=False),'topic_summary.csv','text/csv')
st.subheader('2. Topic word probabilities')
plot_df=pd.DataFrame([{'Topic':f'Topic {i+1}','Word':w,'Probability':p} for i,words in enumerate(top) for w,p in words])
st.plotly_chart(px.bar(plot_df,x='Probability',y='Word',color='Topic',facet_col='Topic',facet_col_wrap=3,orientation='h',height=650),use_container_width=True)
st.subheader('3. Document topic mixtures')
result=df.copy(); result['dominant_topic']=model.theta.argmax(axis=1)+1; result['confidence']=model.theta.max(axis=1)
for i in range(model.n_topics): result[f'Topic {i+1}']=model.theta[:,i]
st.dataframe(result,use_container_width=True)
st.download_button('Download document mixtures CSV',result.to_csv(index=False),'document_topics.csv','text/csv')
st.subheader('4. Topic prevalence')
st.plotly_chart(px.bar(summary,x='Topic',y='Dominant documents'),use_container_width=True)
if 'category' in df.columns:
    st.subheader('5. Category vs. topic')
    counts=pd.crosstab(df.category,result.dominant_topic)
    st.plotly_chart(px.imshow(counts,text_auto=True,aspect='auto',labels={'x':'Dominant topic','y':'Known category'}),use_container_width=True)
    st.json(category_metrics(df.category,result.dominant_topic))
st.caption('Bundled corpus is original synthetic demo text, not the 100 BBC articles from the report. Results will differ from its reported metrics.')
