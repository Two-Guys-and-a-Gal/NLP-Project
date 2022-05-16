import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import json
import requests

import wrangle
import prepare as prep

from env import github_token, github_username

# import for statistical testing
import scipy.stats as stats
from wordcloud import WordCloud

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
df = prep.prep_data(df)
train, validate, test = prep.split_data(df)

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


def vis_one():
    '''
    This funciton......
    '''
    word_counts.sort_values('all', ascending=False).head(5)[['other', 'python', 'r', 'html']].plot.barh()
    plt.title('Word Count for top 5 Most Frequent Overall Words')
    plt.show()
    
def vis_two():
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