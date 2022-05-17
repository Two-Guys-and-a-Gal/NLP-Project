import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import unicodedata
import re
import env
import prepare
from wordcloud import WordCloud
from matplotlib import style

style.use("ggplot")
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")
from collections import Counter


def baseline_accuracy(df, mode):
    """
    Calculate baseline accuracy
    """
    df["mode"] = mode
    baseline_accuracy = accuracy_score(df["actual"], df["mode"])
    return baseline_accuracy


# get the data
df = prepare.prep_data(pd.read_json("data.json"))

# make vectorizer
tfidf = TfidfVectorizer()
# fit the vectorizer to the data and make df
X = tfidf.fit_transform(df["more_clean"])
y = df["language"]

# split data into train and test
X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(X, y)

train = pd.DataFrame(dict(actual=y_train))
validate = pd.DataFrame(dict(actual=y_validate))
test = pd.DataFrame(dict(actual=y_test))


# get mode to use as baseline
mode = df.language.mode().values[0]
# get baseline_accuracy
train_baseline = baseline_accuracy(train, mode)
validate_baseline = baseline_accuracy(validate, mode)
test_baseline = baseline_accuracy(test, mode)


def baseline_accuracy(df, mode):
    """
    Calculate baseline accuracy
    """
    df["mode"] = mode
    baseline_accuracy = accuracy_score(df["actual"], df["mode"])
    return baseline_accuracy


def run_logistic_reg_models():
    """
    Run models on data varying solver and C value
    """
    # get raw data
    df = pd.read_json("data.json")
    # clean data
    df = prepare.prep_data(df)
    # make vectorizer
    tfidf = TfidfVectorizer()
    # fit the vectorizer to the data and make df
    X = tfidf.fit_transform(df["more_clean"])
    y = df["language"]

    # split data into train and test
    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(
        X, y
    )
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    test = pd.DataFrame(dict(actual=y_test))
    # get mode to use as baseline
    mode = df.language.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy(train, mode)
    validate_baseline = baseline_accuracy(validate, mode)
    test_baseline = baseline_accuracy(test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
            "baseline_accuracy": validate_baseline,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = 106  # start at 106 because other models are run first
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for solver in ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]:
        for c in c_values:
            # make the model
            lm = LogisticRegression(C=c, solver=solver).fit(X_train, y_train)
            # run model on data splits
            train["predicted"] = lm.predict(X_train)
            validate["predicted"] = lm.predict(X_validate)
            test["predicted"] = lm.predict(X_test)
            # make results series to add to results df
            stats = pd.Series(
                {
                    "model_number": model_number,
                    "model_type": "LogisticRegression",
                    "solver": solver,
                    "C": c,
                    "train_accuracy": accuracy_score(y_train, train["predicted"]),
                    "validate_accuracy": accuracy_score(
                        y_validate, validate["predicted"]
                    ),
                    "test_accuracy": accuracy_score(y_test, test["predicted"]),
                    "baseline_accuracy": validate_baseline,
                    "better_than_baseline": accuracy_score(
                        y_validate, validate["predicted"]
                    )
                    > validate_baseline,
                }
            )
            # add to results df
            results = pd.concat([results, stats], axis=1)
            model_number += 1

    return results.T.reset_index(drop=True)


def run_decision_tree_models():
    """
    Run models with decision tree classifier
    """
    # get raw data
    df = pd.read_json("data.json")
    df = prepare.prep_data(df)
    # make vectorizer
    tfidf = TfidfVectorizer()
    # fit the vectorizer to the data and make df
    X = tfidf.fit_transform(df["more_clean"])
    y = df["language"]

    # split data into train and test
    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(
        X, y
    )
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    test = pd.DataFrame(dict(actual=y_test))
    # get mode to use as baseline
    mode = df.language.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy(train, mode)
    validate_baseline = baseline_accuracy(validate, mode)
    test_baseline = baseline_accuracy(test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = 1
    max_depths = [1, 2, 3, 4, 5, 10, 100]
    for max_depth in max_depths:
        # make the model
        dtc = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(
            X_train, y_train
        )
        # run model on data splits
        train["predicted"] = dtc.predict(X_train)
        validate["predicted"] = dtc.predict(X_validate)
        test["predicted"] = dtc.predict(X_test)
        # make results series to add to results df
        stats = pd.Series(
            {
                "model_number": model_number,
                "model_type": "DecisionTreeClassifier",
                "max_depth": max_depth,
                "train_accuracy": accuracy_score(y_train, train["predicted"]),
                "validate_accuracy": accuracy_score(y_validate, validate["predicted"]),
                "test_accuracy": accuracy_score(y_test, test["predicted"]),
                "baseline_accuracy": validate_baseline,
                "better_than_baseline": accuracy_score(
                    y_validate, validate["predicted"]
                )
                > validate_baseline,
            }
        )
        # add to results df
        results = pd.concat([results, stats], axis=1)
        model_number += 1

    return results.T.reset_index(drop=True)


def run_random_forest_models():
    """
    Run models with decision tree classifier
    """
    # get raw data
    df = pd.read_json("data.json")
    df = prepare.prep_data(df)
    # make vectorizer
    tfidf = TfidfVectorizer()
    # fit the vectorizer to the data and make df
    X = tfidf.fit_transform(df["more_clean"])
    y = df["language"]

    # split data into train and test
    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare.split_data_xy(
        X, y
    )
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    test = pd.DataFrame(dict(actual=y_test))
    # get mode to use as baseline
    mode = df.language.mode().values[0]
    # get baseline_accuracy
    train_baseline = baseline_accuracy(train, mode)
    validate_baseline = baseline_accuracy(validate, mode)
    test_baseline = baseline_accuracy(test, mode)
    # make a df for results
    results = pd.DataFrame()
    # make baseline model
    baseline_model = pd.Series(
        {
            "model_number": "baseline",
            "model_type": "baseline",
            "train_accuracy": train_baseline,
            "validate_accuracy": validate_baseline,
            "test_accuracy": test_baseline,
            "better_than_baseline": False,
        }
    )
    # add baseline model to results df
    results = pd.concat([results, baseline_model], axis=0)
    # make more models varying solver
    model_number = 8  # start at 8 because of decision tree run first
    max_depths = [1, 2, 3, 4, 5, 10, 100]
    min_sample_leafs = [1, 2, 3, 4, 5, 10, 100]
    criterion = ["gini", "entropy"]
    for max_depth in max_depths:
        for min_samples_leaf in min_sample_leafs:
            for crit in criterion:
                # make the model
                rf = RandomForestClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    criterion=crit,
                ).fit(X_train, y_train)
                # run model on data splits
                train["predicted"] = rf.predict(X_train)
                validate["predicted"] = rf.predict(X_validate)
                test["predicted"] = rf.predict(X_test)
                # make results series to add to results df
                stats = pd.Series(
                    {
                        "model_number": model_number,
                        "model_type": "RandomForest",
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "criterion": crit,
                        "train_accuracy": accuracy_score(y_train, train["predicted"]),
                        "validate_accuracy": accuracy_score(
                            y_validate, validate["predicted"]
                        ),
                        "test_accuracy": accuracy_score(y_test, test["predicted"]),
                        "baseline_accuracy": validate_baseline,
                        "better_than_baseline": accuracy_score(
                            y_validate, validate["predicted"]
                        )
                        > validate_baseline,
                    }
                )
                # add to results df
                results = pd.concat([results, stats], axis=1)
                model_number += 1

    return results.T.reset_index(drop=True)
