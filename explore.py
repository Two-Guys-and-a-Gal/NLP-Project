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
plt.rc('figure', figsize=(13, 7))
# determine figure size
plt.rc('figure', figsize=(20, 8))
# determine font size
plt.rc('font', size=15)
plt.style.use('seaborn-deep')

df = pd.read_json("data.json")
df = prepare.prep_data(df)
train, validate, test = prepare.split_data(df)

def show_counts_and_ratios(df, column):
    """
    Takes in a dataframe and a string of a single column
    Returns a dataframe with absolute value counts and percentage value counts
    """
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    labels
    return labels


def get_common_unique_words(text, threshold=5):
    """
    Get common unique words in dataframe, aka words that occur in multiple readme's
    a word must appear in at least threshold readmes to be considered a common word
    """

    words = text.split()
    counter = Counter(words)
    common_unique_words = [word for word, count in counter.items() if count >= threshold]
    new_string = ' '.join(common_unique_words)
    return new_string


# combine all strings in more_clean where language is the same
languages = train.groupby('language')['more_clean'].apply(lambda x: ' '.join(x)).reset_index()
languages.rename(columns={'more_clean':'all_words'}, inplace=True)
languages['unique_words'] = train.groupby('language')['unique_words'].apply(lambda x: ' '.join(x)).values
languages['common_unique_words'] = languages.unique_words.apply(get_common_unique_words)
languages['n_words'] = languages['all_words'].apply(lambda x: len(x.split()))
languages['unique_word_count'] = languages['all_words'].apply(lambda x: len(set(x.split())))
languages['mean_word_count'] = train.groupby('language')['word_count'].mean().values.round(1)
languages['median_word_count'] = train.groupby('language')['word_count'].median().values.round(1)
languages['most_common_word'] = languages['unique_words'].apply(lambda x: prepare.n_most_common_word(x))
languages['2nd_most_common_word'] = languages['unique_words'].apply(lambda x: prepare.n_most_common_word(x,2))
languages['3rd_most_common_word'] = languages['unique_words'].apply(lambda x: prepare.n_most_common_word(x,3))
languages['4th_most_common_word'] = languages['unique_words'].apply(lambda x: prepare.n_most_common_word(x,4))
languages['5th_most_common_word'] = languages['unique_words'].apply(lambda x: prepare.n_most_common_word(x,5))




# makes sets of all words in all languages so no words are repeated
html_set = set(languages[languages.language=='HTML'].common_unique_words.values[0].split())
python_set = set(languages[languages.language=='Python'].common_unique_words.values[0].split())
r_set = set(languages[languages.language=='R'].common_unique_words.values[0].split())
other_set = set(languages[languages.language=='Other'].common_unique_words.values[0].split())
#remove words found in other languages
unique_to_html = " ".join(html_set - python_set - r_set - other_set)
unique_to_python = " ".join(python_set - html_set - r_set - other_set)
unique_to_r = " ".join(r_set - html_set - python_set - other_set)
unique_to_other = " ".join(other_set - html_set - python_set - r_set)
# make a series to add to the dataframe
unique_to_lang = [unique_to_html, unique_to_other, unique_to_python, unique_to_r]
languages['unique_to_language'] = unique_to_lang




## Create lists of words for each language category
other = (' '.join(train[train.language == 'Other'].more_clean)).split()
python = (' '.join(train[train.language == 'Python'].more_clean)).split()
r = (' '.join(train[train.language == 'R'].more_clean)).split()
html = (' '.join(train[train.language == 'HTML'].more_clean)).split()
all_words = (' '.join(train.more_clean)).split()


## Transform lists into series
other_freq = pd.Series(other).value_counts()
python_freq = pd.Series(python).value_counts()
r_freq = pd.Series(r).value_counts()
html_freq = pd.Series(html).value_counts()
all_freq = pd.Series(all_words).value_counts()


#Create a word_counts data frame we can work with

word_counts = (pd.concat([all_freq, other_freq, python_freq, r_freq, html_freq], axis=1, sort=True)
                .set_axis(['all', 'other', 'python', 'r', 'html'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))


def vis_one_a():
    '''
    This funciton......
    '''
    word_counts.sort_values('all', ascending=False).head(5)[['other', 'python', 'r', 'html']].plot.barh()
    plt.title('Word Count for top 5 Most Frequent Overall Words')
    plt.show()
    
def vis_one_b():
    (word_counts.sort_values('all', ascending=False)
     .head(10)
     .apply(lambda row: row/row['all'], axis = 1)
     .drop(columns = 'all')
     .sort_values(by = 'other')
     .plot.barh(stacked = True, width = 1, ec = 'k')
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('% of Top 10 Word Frequency by Language')
    plt.show()
    
    
def vis_two():
    # graph number of words in unique_to_language
    new_df= languages[['language','unique_to_language']]
    #new_df.set_index('language', inplace=True)
    new_df['unique_words'] = new_df['unique_to_language'].apply(lambda x: len(x.split()))
    new_df.sort_values(by='unique_words', ascending=False, inplace=True)
    new_df.plot.bar(x='language', y='unique_words')
    plt.title('Number of Words Unique to each Language')
    plt.legend().set_visible(False)
    # determine figure size
    plt.rc('figure', figsize=(20, 8))
    # determine font size
    plt.rc('font', size=15)
    # determine style
    plt.style.use('seaborn-deep')
    plt.show()
    
    
def vis_five():
    ax = sns.barplot(data=train, y='word_count', x='language', ci=None)
    ax.set(title = 'Average README Word Count by Programming Language', xlabel='Top 4 Programming Languages', ylabel='Word Count')
    plt.show()
    
def mann_whitney():
    # Establish variables for statistical testing
    all_but_r = train[train.language != 'R'].word_count
    r = train[train.language == 'R'].word_count

    # Set Alpha
    alpha = 0.05

    # Run Mann Whitney Test
    stat, p = stats.mannwhitneyu(all_but_r, r, method="exact")
    # Obtain test statistics through a print statement
    print(f'Mann Whitney Test Statistics: Statistic  {stat}, P Value {p}')

    # Evaluate outcome of statistical testing
    if p < alpha:
        print("We reject $H_{0}$")
    else:
        print("We fail to reject $H_{0}$")