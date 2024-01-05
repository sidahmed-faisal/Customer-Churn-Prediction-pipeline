# library doc string
"""
Modules of chrun customer analysis.

Author : Sidahmed Faisal

Date : 14 January 2024
"""
# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import seaborn as sns
sns.set()

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    Return dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    Perform eda on df and save figures to images folder.
    input:
            df: pandas dataframe

    output:
            image: a plot based of the dataframe
    """
    # encoding churn column
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    # making a histogram plot
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    # saving plot
    plt.savefig("./images/eda/histplot.png")
    return df


def encoder_helper(df, category_lst, response):
    """
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    # loop through categorical columns and process each one
    for category in category_lst:
        column_lst = []
        column_group = df.groupby(category).mean()[response]
        for val in df[category]:
            column_lst.append(column_group.loc[val])
        column_name = category + "_" + response
        df[column_name] = column_lst
    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # categorical features
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    # feature engineering
    encoded_df = encoder_helper(
        df, cat_columns, response
    )
    # target feature
    y = encoded_df["Churn"]
    # Create dataframe
    X = pd.DataFrame()
    # adding response string to each categorical column
    for i, val in enumerate(cat_columns):
        cat_columns[i] = val + "_" + response
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]
    keep_cols.extend(cat_columns)
    # Features DataFrame
    X[keep_cols] = encoded_df[keep_cols]
    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return (X_train, X_test, y_train, y_test)


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder.

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             Saved Reports images
    """
    # Random Forest
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig(fname="./images/results/rf_results.png")

    # Logistic Regression
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(fname="./images/results/Lg_results.png")


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores the feature importances in pth.
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort Feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname="./images/results/feature_importances.png")


def train_models(X_train, X_test, y_train, y_test):
    """
    Train, store model results: images + scores, and store models.
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # Random Forest model
    rfc = RandomForestClassifier(random_state=42)

    # Logistic Regression model
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    # grid search for random forest parameters and instantiation
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train Ramdom Forest using GridSearch
    cv_rfc.fit(X_train, y_train)

    # Train Logistic Regression
    lrc.fit(X_train, y_train)

    # Save Models
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Create Classification Report
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8
    )
    plt.savefig(fname="./images/results/roc_curve_result.png")

    # Display feature importance on train data
    feature_importance_plot(
        cv_rfc, X_train, "./images/results"
    )


if __name__ == "__main__":
    # Import data
    BANK_DF = import_data(pth="./data/bank_data.csv")

    # Perform EDA
    EDA = perform_eda(BANK_DF)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        EDA, "Churn"
    )

    # Trainning start
    print("trainning has started")

    # Model training,prediction and evaluation
    train_models(
        X_train=X_TRAIN,
        X_test=X_TEST,
        y_train=Y_TRAIN,
        y_test=Y_TEST)
