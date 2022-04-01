# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: 'Python 3.10.0 64-bit (''cmpe-351-_rWzjxJw'': pipenv)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMPE 351 Exercise 3
#
# Analysis will be done on the title.

# %% [markdown]
#
# ## Setting up


# %%
# Resolving paths in a platform agnostic way.
import logging
import multiprocessing
import pickle
from os.path import dirname, join, realpath
from pathlib import Path
from string import punctuation
from time import time

import nltk
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from spacy.lang.en import English


# %%
def is_interactive():
    """Check if the script is being run interactively."""
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    SCRIPT_DIR = dirname(realpath("__file__"))
else:
    SCRIPT_DIR = dirname(realpath(__file__))

DATA_DIR = join(SCRIPT_DIR, "data")
MODELS_DIR = join(SCRIPT_DIR, "models")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# %%
raw_questions_df = pd.read_csv(join(DATA_DIR, "questions.csv"))
raw_questions_df.head()

# %%
pd.set_option("display.max_colwidth", None)

# %%
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# %%
logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# %%
stop_words = set(stopwords.words("english"))

# %% [markdown]
# ## Data Preprocessing

# %%
question_titles_df = raw_questions_df["Title"]
question_titles_df.head()

# %% [markdown]
# Lower casing the text.

# %%
question_titles_df = question_titles_df.str.lower()
question_titles_df.head()

# %% [markdown]
# Removing Punctuation

# %%
PUNCTUATION_TO_REMOVE = punctuation


def remove_punctuation(text: str):
    """Remove punctuation form a string."""
    return text.translate(str.maketrans("", "", PUNCTUATION_TO_REMOVE))


question_titles_df = question_titles_df.apply(remove_punctuation)
question_titles_df.head()


# %% [markdown]
# Removing stop words

# %%
def remove_stopwords(text: str):
    """Remove stopwords from a string."""
    return " ".join(
        [word for word in str(text).split() if word not in stop_words]
    )


question_titles_df = question_titles_df.apply(remove_stopwords)
question_titles_df.head()

# %% [markdown]
# Stemming the words.

# %%
stemmer = PorterStemmer()


def stem_words(text: str):
    """Stem words in a string."""
    return " ".join([stemmer.stem(word) for word in text.split()])


question_titles_df = question_titles_df.apply(stem_words)
question_titles_df.head()

# %% [markdown]
# Tokenizing the text.

# %%
question_titles_df = question_titles_df.apply(nltk.word_tokenize)
question_titles_df.head()

# %% [markdown]
# ## Learning word vectors from text corpus

# %%
cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(
    min_count=1,
    workers=cores - 1,
)

# %%
t = time()
w2v_model.build_vocab(question_titles_df, progress_per=10000)
print("Time to build vocab: {} mins".format(round((time() - t) / 60, 2)))

# %%
t = time()
w2v_model.train(
    question_titles_df,
    total_examples=w2v_model.corpus_count,
    epochs=1,
    report_delay=1,
)
print("Time to train the model: {} mins".format(round((time() - t) / 60, 2)))

# %%
w2v_model.save(join(MODELS_DIR, "word2vec.model"))

# %% [markdown]
# ## Topic Modelling

# %%
question_titles_df = raw_questions_df["Title"]
question_titles_df.head()

# %%
parser = English()


def tokenize(text: str):
    lda_tokens: list[str] = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        if token.like_url:
            lda_tokens.append("URL")
        elif token.orth_.startswith("@"):
            lda_tokens.append("SCREEN_NAME")
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


# %%
def get_lemma(word: str):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    return lemma


# %%
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# %%
tokenized_question_titles = question_titles_df.apply(prepare_text_for_lda)
tokenized_question_titles.head()

# %%
dictionary = Dictionary(documents=tokenized_question_titles)
corpus = tokenized_question_titles.apply(dictionary.doc2bow)
pickle.dump(corpus, open(join(MODELS_DIR, "corpus.pkl"), "wb"))
dictionary.save(join(MODELS_DIR, "dictionary.gensim"))

# %% [markdown]
# > Briefly specify how you [picked] the number of topics.
#
# I picked 5 as the number of topics because it was the first number used in [this tutorial I followed](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21#afa6).

# %%
NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary)
ldamodel.save(join(MODELS_DIR, "lda_model_5.gensim"))
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# %% [markdown]
# ## Summmary

# %%
dictionary = Dictionary.load(join(MODELS_DIR, "dictionary.gensim"))
corpus = pickle.load(open(join(MODELS_DIR, "corpus.pkl"), "rb"))
lda = LdaModel.load(join(MODELS_DIR, "lda_model_5.gensim"))
lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
