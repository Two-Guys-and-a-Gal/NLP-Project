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


def get_unique_words(text):
    """
    Get unique words in dataframe
    """

    # Create a list of words separated by a space
    words = text.split()
    # Create a variable to hold a list of unique words
    unique_words = set(words)
    # Join unique words list with a new string
    new_string = " ".join(unique_words)
    return new_string


def get_unique_word_count(string):
    """
    This function will take in a string and return the number of unique words in that string.
    """

    # Create a list of words separated by a space
    words = string.split()
    # Create a variable to hold a list of unique words
    words = set(words)
    # Calculate and return the length of the list

    return len(words)


def get_sentence_count(string):
    """
    This function will take in a string and return the number of sentences in that string.
    """
    # Create a list of sentences using nltk
    sentences = nltk.sent_tokenize(string)
    # Calculate and return length of the list
    return len(sentences)


def n_most_common_word(string, n=1):
    """
    Return the nth most common word in a string
    """
    # Create a list of words separated by a space
    words = string.split()
    # Make an if statement that will only show the nth most common word based on the value set for n
    if len(words) < n:
        return ""
    # Use collections to get an ngram count
    word_counts = Counter(words)
    # Return only the most common
    return word_counts.most_common(n)[n - 1][0]


##### Cleaning Functions ##############


def basic_clean(text):
    """
    This function will take in a string and perform basic cleaning functions. It reduces all characters
    to lower case, normalizes unicode characters, and removes anything that is not a letter or whitespace. 
    """
    # Normalize unicode characters
    text = (
        unicodedata.normalize("NFKD", text)
        .encode(
            "ascii", "ignore"
        )  # encode into ascii byte strings, and ingnore unknown characers.
        .decode("utf-8", "ignore")  # decode back into a workable utf-8 string
    )
    text = re.sub(
        r"[^a-zA-Z\s]", "", text
    ).lower()  # Removes anything not a letter or white space and lowercases everything
    return text


def tokenize(text):
    """
    This function will take in a string, tokenize all words and return the tokenized string.
    """
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def stem(text):
    """
    This function will take in a string and and return a stemmed version of that text. 
    """
    # Create the stemmer
    stemmer = nltk.PorterStemmer()
    # Use the stemmer to stem the text
    stems = [stemmer.stem(token) for token in text.split()]
    # Join the string w
    string = " ".join(stems)
    return string


def lemmatize(text):
    """
    This function will take in a tokenized string and return a lemmatized version. 
    """
    # Create the lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # Use the lemmatizer to lemmatize the text
    lemmas = [lemmatizer.lemmatize(token) for token in text.split()]
    string = " ".join(lemmas)
    return string


def remove_stopwords(text, extra_stopwords=[], exclude_stopwords=[]):
    """
    Remove stopwords from text
    """
    # Create a list of stopwords
    stop_words = stopwords.words("english")
    # Add extra stopwords to the list
    stop_words = set(stop_words).union(set(extra_stopwords))
    # Remove stopwords from the text
    stop_words = set(stop_words) - set(exclude_stopwords)
    # resplit the text
    words = text.split()
    # filter out the stopwords
    filtered_words = [word for word in words if word not in stop_words]
    # join the filtered words
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
        lambda x: "Other" if x not in top_language_counts else x
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

    # drop rows with nulls
    df = df.dropna(axis=0)
    # remove rows with a zero word count
    df = df[df["readme_contents"].str.len() > 0]

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
        df["unique_word_count"] = df.more_clean.apply(get_unique_word_count)
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


def split_data(df):

    """
    This function takes in a dataframe, then splits and returns the data as train, validate, and test sets 
    using random state 123.
    """
    # split data into 2 groups, train_validate and test, assigning test as 20% of the dataset
    train_validate, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["language"]
    )
    # split train_validate into 2 groups with
    train, validate = train_test_split(
        train_validate,
        test_size=0.3,
        random_state=42,
        stratify=train_validate["language"],
    )
    return train, validate, test


def split_data_xy(X, y):
    """
    This function takes in X and y variables as strings, then splits and returns the data as 
    X_train, X_validate, X_test, y_train, y_validate, and y_test sets using random state 42.
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
