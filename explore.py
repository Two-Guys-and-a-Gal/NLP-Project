import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import json
import requests

import wrangle
import prepare

from env import github_token, github_username

# import for statistical testing
import scipy.stats as stats
from wordcloud import WordCloud
from collections import Counter

from sklearn.model_selection import train_test_split

import re
import unicodedata
import nltk

# Set Universal Visualization Formatting
plt.rc("figure", figsize=(13, 7))
# determine figure size
plt.rc("figure", figsize=(20, 8))
# determine font size
plt.rc("font", size=15)
plt.style.use("seaborn-deep")

df = pd.read_json("data.json")
df = prepare.prep_data(df)
train, validate, test = prepare.split_data(df)


def show_counts_and_ratios(df, column):
    """
    Takes in a dataframe and a string of a single column
    Returns a dataframe with absolute value counts and percentage value counts
    """
    labels = pd.concat(
        [df[column].value_counts(), df[column].value_counts(normalize=True)], axis=1
    )
    labels.columns = ["n", "percent"]
    labels
    return labels


def get_common_unique_words(text, threshold=5):
    """
    Get common unique words in dataframe, aka words that occur in multiple readme's
    a word must appear in at least threshold readmes to be considered a common word
    """

    words = text.split()
    counter = Counter(words)
    common_unique_words = [
        word for word, count in counter.items() if count >= threshold
    ]
    new_string = " ".join(common_unique_words)
    return new_string


# combine all strings in more_clean where language is the same
languages = (
    train.groupby("language")["more_clean"].apply(lambda x: " ".join(x)).reset_index()
)
languages.rename(columns={"more_clean": "all_words"}, inplace=True)
languages["unique_words"] = (
    train.groupby("language")["unique_words"].apply(lambda x: " ".join(x)).values
)
languages["common_unique_words"] = languages.unique_words.apply(get_common_unique_words)
languages["n_words"] = languages["all_words"].apply(lambda x: len(x.split()))
languages["unique_word_count"] = languages["all_words"].apply(
    lambda x: len(set(x.split()))
)
languages["mean_word_count"] = (
    train.groupby("language")["word_count"].mean().values.round(1)
)
languages["median_word_count"] = (
    train.groupby("language")["word_count"].median().values.round(1)
)
languages["most_common_word"] = languages["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x)
)
languages["2nd_most_common_word"] = languages["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 2)
)
languages["3rd_most_common_word"] = languages["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 3)
)
languages["4th_most_common_word"] = languages["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 4)
)
languages["5th_most_common_word"] = languages["unique_words"].apply(
    lambda x: prepare.n_most_common_word(x, 5)
)


def get_common_bigrams(text, threshold=5):
    """
    This function takes in a text and returns a list of the top 5 bigrams that are common to the text.
    """
    # get all bigrams in text
    bigrams = pd.Series(nltk.ngrams(text.split(), 2)).value_counts()
    # filter out bigrams that are less than threshold
    bigrams = bigrams[bigrams > threshold]
    # return all that occur more than 5 times
    return bigrams.index.values


languages["common_bigrams"] = languages.all_words.apply(lambda x: get_common_bigrams(x))


# makes sets of all words in all languages so no words are repeated
html_set = set(
    languages[languages.language == "HTML"].common_unique_words.values[0].split()
)
python_set = set(
    languages[languages.language == "Python"].common_unique_words.values[0].split()
)
r_set = set(languages[languages.language == "R"].common_unique_words.values[0].split())
other_set = set(
    languages[languages.language == "Other"].common_unique_words.values[0].split()
)
# remove words found in other languages
unique_to_html = " ".join(html_set - python_set - r_set - other_set)
unique_to_python = " ".join(python_set - html_set - r_set - other_set)
unique_to_r = " ".join(r_set - html_set - python_set - other_set)
unique_to_other = " ".join(other_set - html_set - python_set - r_set)
# make a series to add to the dataframe
unique_to_lang = [unique_to_html, unique_to_other, unique_to_python, unique_to_r]
languages["unique_to_language"] = unique_to_lang
# add bigrams to language dataframe
languages["bigrams"] = languages.all_words.apply(
    lambda x: pd.Series(nltk.ngrams(x.split(), 2)).values
)

# makes sets of all common bigrams in all languages so no bigrams are repeated
html_set = set(languages[languages.language == "HTML"].common_bigrams.values[0])
python_set = set(languages[languages.language == "Python"].common_bigrams.values[0])
r_set = set(languages[languages.language == "R"].common_bigrams.values[0])
other_set = set(languages[languages.language == "Other"].common_bigrams.values[0])
# remove words found in other languages
unique_to_html = html_set - python_set - r_set - other_set
unique_to_python = python_set - html_set - r_set - other_set
unique_to_r = r_set - html_set - python_set - other_set
unique_to_other = other_set - html_set - python_set - r_set
# make a series to add to the dataframe
unique_to_lang = [unique_to_html, unique_to_other, unique_to_python, unique_to_r]
languages["bigrams_unique_to_language"] = list(unique_to_lang)


## Create lists of words for each language category
other = (" ".join(train[train.language == "Other"].more_clean)).split()
python = (" ".join(train[train.language == "Python"].more_clean)).split()
r = (" ".join(train[train.language == "R"].more_clean)).split()
html = (" ".join(train[train.language == "HTML"].more_clean)).split()
all_words = (" ".join(train.more_clean)).split()


## Transform lists into series
other_freq = pd.Series(other).value_counts()
python_freq = pd.Series(python).value_counts()
r_freq = pd.Series(r).value_counts()
html_freq = pd.Series(html).value_counts()
all_freq = pd.Series(all_words).value_counts()

# make bigrams
top_5_other_bigrams = pd.Series(nltk.ngrams(other, 2)).value_counts().head(5)
top_5_python_bigrams = pd.Series(nltk.ngrams(python, 2)).value_counts().head(5)
top_5_r_bigrams = pd.Series(nltk.ngrams(r, 2)).value_counts().head(5)
top_5_html_bigrams = pd.Series(nltk.ngrams(html, 2)).value_counts().head(5)


# Create a word_counts data frame we can work with

word_counts = (
    pd.concat([all_freq, other_freq, python_freq, r_freq, html_freq], axis=1, sort=True)
    .set_axis(["all", "other", "python", "r", "html"], axis=1, inplace=False)
    .fillna(0)
    .apply(lambda s: s.astype(int))
)


def vis_one_a():
    """
    This function creates a visualization for the top 5 most frequent words overall across the top
    4 coding languages. It calculates the value_counts for word_counts by language then plots a
    horizontal bar graph.
    """
    word_counts.sort_values("all", ascending=False).head(5)[
        ["other", "python", "r", "html"]
    ].plot.barh()
    plt.title("Word Count for top 5 Most Frequent Overall Words")
    plt.show()


def vis_one_b():
    """
    This function creates a visualization for the % of Top 10 Word Frequency by Language across the top
    4 coding languages. It plots a horizontal staked bar graph.
    """
    (
        word_counts.sort_values("all", ascending=False)
        .head(10)
        .apply(lambda row: row / row["all"], axis=1)
        .drop(columns="all")
        .sort_values(by="other")
        .plot.barh(stacked=True, width=1, ec="k")
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("% of Top 10 Word Frequency by Language")
    plt.show()


def vis_two():
    """
    This function creates a visualization for the number of words unique to each of the top 4 programming languages. 
    It plots a bar graph with the number of unique words for each of the languages.
    """
    # graph number of words in unique_to_language
    new_df = languages[["language", "unique_to_language"]]
    # new_df.set_index('language', inplace=True)
    new_df["unique_words"] = new_df["unique_to_language"].apply(
        lambda x: len(x.split())
    )
    new_df.sort_values(by="unique_words", ascending=False, inplace=True)
    new_df.plot.bar(
        x="language",
        y="unique_words",
        color=["seagreen", "steelblue", "brown", "slateblue"],
    )
    plt.title("Number of Words Unique to each Language")
    plt.legend().set_visible(False)
    # determine figure size
    plt.rc("figure", figsize=(20, 8))
    # determine font size
    plt.rc("font", size=15)
    # determine style
    plt.style.use("seaborn-deep")
    plt.show()


def vis_three():
    """
    This function creates a visualization for the number of unique bigrams to each of the top 4 programming languages. 
    It plots a bar graph with the number of unique bigrams for each of the languages.
    """
    # graph number of bigrams in unique_to_language
    new_df = languages[["language", "bigrams_unique_to_language"]]
    # new_df.set_index('language', inplace=True)
    new_df["unique_bigrams"] = new_df["bigrams_unique_to_language"].apply(
        lambda x: len(x)
    )
    new_df.sort_values(by="unique_bigrams", ascending=False, inplace=True)
    new_df.plot.bar(
        x="language",
        y="unique_bigrams",
        color=["brown", "seagreen", "steelblue", "slateblue"],
    )
    plt.title("Number of Common Bigrams Unique to each Language")
    plt.legend().set_visible(False)
    plt.show()


def vis_four():
    """
    This function creates a visualization for the top 5 bigrams for each of the 4 programming languages. 
    It plots 4 subplots, to display a bar graph for each language with the top five bigrams for each 
    of the languages.
    """
    ## Plot Top 5 Bigrams
    fig, axs = plt.subplots(2, 2)
    # set figure size
    fig.set_size_inches(35, 12)
    # set title
    plt.suptitle("Top 5 Bigrams by Programming Language", fontsize=24)
    top_5_other_bigrams.sort_values().plot.barh(
        color="steelblue", width=0.9, ax=axs[0, 0]
    )
    axs[0, 0].set_title("Other")
    # axs[0,0].set_ylabel('Bigram')

    top_5_python_bigrams.sort_values().plot.barh(
        color="seagreen", width=0.9, ax=axs[0, 1]
    )
    axs[0, 1].set_title("Python")

    top_5_html_bigrams.sort_values().plot.barh(color="brown", width=0.9, ax=axs[1, 0])
    axs[1, 0].set_title("HTML")

    top_5_r_bigrams.sort_values().plot.barh(color="slateblue", width=0.9, ax=axs[1, 1])
    axs[1, 1].set_title("R")
    plt.show()


def vis_five():
    """
    This function creates a average number of words in a README by top 4 programming language. 
    It plots a bar graph with the average number of words by programming language.
    """
    ax = sns.barplot(data=train, y="word_count", x="language", ci=None)
    ax.set(
        title="Average README Word Count by Programming Language",
        xlabel="Top 4 Programming Languages",
        ylabel="Word Count",
    )
    plt.show()


def mann_whitney():
    """
    This function conducts a Mann Whitney hypothesis test on the variables established inside the function. 
    It displays the statistics for the test results and evaluates the results against the Null hypothesis. 
    """
    # Establish variables for statistical testing
    all_but_r = train[train.language != "R"].word_count
    r = train[train.language == "R"].word_count

    # Set Alpha
    alpha = 0.05

    # Run Mann Whitney Test
    stat, p = stats.mannwhitneyu(all_but_r, r, method="exact")
    # Obtain test statistics through a print statement
    print(f"Mann Whitney Test Statistics: Statistic  {stat}, P Value {p}")

    # Evaluate outcome of statistical testing
    if p < alpha:
        print("We reject $H_{0}$")
    else:
        print("We fail to reject $H_{0}$")
