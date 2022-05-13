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

def prep_data(df, extra_stopwords=[], exclude_stopwords=[], remove_jupyter = False, keep_top_languages = True):
    '''
    This function take in a df with 
    option to pass lists for extra_words and exclude_words and option to 
    remove all rows with Jupyter Notebook in the language column. It
    returns a df with the original readme_contents, a cleaned version and 
    more_clean version that has been lemmatized with stopwords removed.
    '''
    if remove_jupyter:
        df = df[df['language'] != 'Jupyter Notebook'].copy()
        
    if keep_top_languages:
        df = keep_top_n_languages(df).copy()
    
    df.rename(columns={'readme_contents':'original'}, inplace=True)
    
    df['clean'] = df['original'].apply(basic_clean)\
                            .apply(remove_stopwords,
                                  extra_stopwords=extra_stopwords,
                                  exclude_stopwords=exclude_stopwords)
    
    df['more_clean'] = df['clean'].apply(lemmatize)
    
    return df

def split_data(X, y):
    """ 
    General use function to split data into train and test sets. 
    """
    # split the data set with stratifiy if True
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.3, random_state=42
    )
    return (X_train, X_validate, X_test, y_train, y_validate, y_test)


# obselete
