# LDA Topic Modeling — Lab Exercise 4

A runnable reconstruction inspired by the uploaded lab report. Streamlit app with text preprocessing, bag-of-words, scikit-learn LDA, top-five topic words, per-document topic mixtures, charts, category comparison, CSV exports, and an evaluation CLI.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **Discover Topics**. Change **number of topics**, **random seed**, or **corpus** to obtain different outputs. Paste multiple documents separated by blank lines, or upload TXT/CSV.

## Generate output files

```bash
python -m eval.run_eval
pytest -q
```

This writes `outputs/top_words.csv`, `outputs/document_topics.csv`, and `outputs/summary.json`.

## Important dataset note

The uploaded PDF describes a 100-document sample of the BBC News dataset but does not contain the original 100 full articles. This package instead bundles **100 original synthetic news-style demonstration documents** across five categories, with 20 documents each. It does **not** claim to reproduce the report's 677-word vocabulary, topic weights, coherence, or evaluation metrics. To reproduce the reported experiment, supply the original BBC sample CSV through the app. NLTK WordNet lemmatization is optional and skipped if the WordNet corpus is not installed. To enable it: `python -m nltk.downloader wordnet`.

## Structure

- `app.py`: Streamlit UI
- `src/preprocess.py`: tokenization, stop words, optional lemmatization
- `src/topic_model.py`: count vectorizer, LDA multi-restart fitting, top words
- `src/metrics.py`: purity, NMI, ARI
- `data/demo_news.csv`: synthetic demonstration dataset
- `eval/run_eval.py`: reproducible demo evaluation and CSV outputs
- `tests/test_pipeline.py`: basic pipeline test
- `outputs/`: generated evaluation files

## Suggested demos

1. K=5, seed=42: five-topic overview.
2. K=3, seed=42: topics may merge.
3. K=7, seed=42: topics may split.
4. K=5, seed=11: observe sensitivity to initialization.
5. Paste your own text to see entirely different topic-word and document-mixture outputs.

Topic numbers have no inherent semantic labels; inspect top words before interpreting them.
