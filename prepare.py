import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pandas as pd
from time import strftime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def basic_clean(text):
    """
    Basic cleaning of text
    """
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    text = re.sub(r"[^\w\s]", " ", text).lower()
    return text


def tokenize(text):
    """
    Tokenize text
    """
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def stem(tokens, use_tokens=False):
    """
    Stem tokens
    """
    stemmer = nltk.PorterStemmer()
    if use_tokens:
        stems = [stemmer.stem(token) for token in tokens]
    else:
        stems = [stemmer.stem(token) for token in tokens.split()]
    string = " ".join(stems)
    return string


def lemmatize(tokens, use_tokens=False):
    """
    Lemmatize tokens
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    if use_tokens:
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        lemmas = [lemmatizer.lemmatize(token) for token in tokens.split()]
    string = " ".join(lemmas)
    return string


def remove_stopwords(
    tokens, extra_stopwords=[], exclude_stopwords=[], use_tokens=False
):
    """
    Remove stopwords from tokens
    """
    stop_words = stopwords.words("english")
    stop_words = set(stop_words).union(set(extra_stopwords))
    stop_words = set(stop_words) - set(exclude_stopwords)
    if use_tokens:
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    else:
        words = tokens.split()
        filtered_words = [word for word in words if word not in stop_words]
        string_without_stopwords = " ".join(filtered_words)
        return string_without_stopwords


def more_clean(text):
    """
    More cleaning of text
    """
    text = basic_clean(text)
    text = lemmatize(text)
    text = remove_stopwords(text)
    return text


def keep_top_n_languages(df, n_languages=3):
    """
    Get the top language counts and make all non-top languages other
    """
    # get the top language counts
    top_language_counts = df["language"].value_counts().head(n_languages).index.tolist()
    # make all non-top languages 'other'
    df["language"] = df["language"].apply(
        lambda x: "other" if x not in top_language_counts else x
    )
    return df


def split_data(df, y_value, stratify=True):
    """ General use function to split data into train and test sets. 
    Stratify = True is helpful for categorical y values"""
    # split the data set with stratifiy if True
    if stratify:
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[y_value]
        )
        train, validate = train_test_split(
            train, test_size=0.3, random_state=42, stratify=train[y_value]
        )
    else:  # if stratify is false (for non-categorical y values)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train, validate = train_test_split(train, test_size=0.3, random_state=42)
    return (train, validate, test)


def split_x_y(df, y_value):
    """split data into x and y"""
    x = df.drop(columns=[y_value])
    y = df[y_value]
    return x, y


def get_splits(df, y_value, stratify=True):
    """
    Get splits for train, validate, and test
    """
    train, validate, test = split_data(df, y_value, stratify=stratify)
    x_train, y_train = split_x_y(train, y_value)
    x_validate, y_validate = split_x_y(validate, y_value)
    x_test, y_test = split_x_y(test, y_value)
    return (x_train, y_train, x_validate, y_validate, x_test, y_test)

