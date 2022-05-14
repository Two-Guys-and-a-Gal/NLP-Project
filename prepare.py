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
from collections import Counter


###### Functions to add new Features ##########


def get_char_count(string):
    """
    This function will take in a string and return the number of characters in it.
    """

    return len(string)


def get_word_count(string):
    """
    This function will take in a string and return the number of words in that string.
    This function will include repeat words.
    """

    # Create a list of words separated by a space
    words = string.split()

    return len(words)


def get_unique_words(string):
    """
    This function will take in a string and return the number of unique words in that string.
    """

    words = string.split()
    words = set(words)

    return len(words)


def get_sentence_count(string):
    """
    This function will take in a string and return the number of sentences in that string.
    """

    sentences = nltk.sent_tokenize(string)

    return len(sentences)


def n_most_common_word(string, n=1):
    """
    Return the nth most common word in a string
    """
    words = string.split()
    if len(words) < n:
        return ""
    word_counts = Counter(words)
    return word_counts.most_common(n)[n - 1][0]


##### Cleaning Functions ##############


def basic_clean(text):
    """
    Basic cleaning of text
    """
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    ##text = re.sub(r"[^\w\s]", " ", text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
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


def prep_data(
    df,
    extra_stopwords=[],
    exclude_stopwords=[],
    keep_top_languages=True,
    add_features=True,
):
    """
    This function take in a df with 
    option to pass lists for extra_words and exclude_words and option to 
    remove all rows with Jupyter Notebook in the language column. It
    returns a df with the original readme_contents, a cleaned version and 
    more_clean version that has been lemmatized with stopwords removed.
    """

    # must be above the 'keep_top_languages' option
    # we manually looked in the data and found the jupyter notebooks were python
    df["language"].replace({"Jupyter Notebook": "Python"}, inplace=True)

    # takes top 3 languages and makes all others 'other'
    if keep_top_languages:
        df = keep_top_n_languages(df).copy()

    # change name to make more readible
    df.rename(columns={"readme_contents": "original"}, inplace=True)

    df["more_clean"] = (
        df["original"]
        .apply(basic_clean)
        .apply(
            remove_stopwords,
            extra_stopwords=extra_stopwords,
            exclude_stopwords=exclude_stopwords,
        )
        .apply(lemmatize)
    )

    # df["more_clean"] = df["clean"].apply(lemmatize)

    if add_features:
        df["unique_words"] = df["more_clean"].apply(get_unique_words)
        df["char_count"] = df.more_clean.apply(get_char_count)
        df["word_count"] = df.more_clean.apply(get_word_count)
        df["unique_word_count"] = df.more_clean.apply(get_unique_words)
        # add column to df with most common word
        df["most_common_word"] = df["more_clean"].apply(n_most_common_word)
        # add column to df with 2nd common word
        df["2nd_most_common_word"] = df["more_clean"].apply(n_most_common_word, n=2)
        # add column to df with 3rd common word
        df["3rd_most_common_word"] = df["more_clean"].apply(n_most_common_word, n=3)
        # add column to df with 4th common word
        df["4th_most_common_word"] = df["more_clean"].apply(n_most_common_word, n=4)
        # add column to df with 5th common word
        df["5th_most_common_word"] = df["more_clean"].apply(n_most_common_word, n=5)

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
