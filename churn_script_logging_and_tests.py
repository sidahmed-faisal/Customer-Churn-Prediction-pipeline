# library doc string
"""
Modules of chrun customer logging and Testing.

Author : Sidahmed Faisal

Date : 14 January 2024
"""
# import libraries
import os
import logging
import churn_library as ch

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = ch.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    Test perform eda function
    '''
    dataframe = ch.import_data("./data/bank_data.csv")
    try:
        ch.perform_eda(dataframe)
        logging.info("SUCCESS: DataFrame was loaded ")
    except KeyError as err:
        logging.error("FAILED: DataFrame wasn't found")
    # Assert if customer_age distribution is saved
    try:
        assert os.path.isfile("./images/eda/histplot.png") is True
        logging.info('SUCCESS: Customers age distributions File was saved')
    except AssertionError as err:
        logging.error('FAILED: File wasn\'t saved ')
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper
    '''
    # Load DataFrame
    dataframe = ch.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Categorical Features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        encoded_df = ch.encoder_helper(
            dataframe,
            cat_columns,
            'Churn')

        # Assert new Columns were added
        assert len(encoded_df.select_dtypes(include='object').columns) > 0

        logging.info("SUCCESS: Categorical Columns have been added")
    except AssertionError as err:
        logging.error("FAILED: Categorical Columns were not added")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering
    '''
    # Load the DataFrame
    dataframe = ch.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    try:
        X_train, X_test, y_train, y_test = ch.perform_feature_engineering(
            dataframe,
            'Churn')

        # Assert that data has been splitted successfully
        assert int(len(dataframe) * 0.3 + 1) == len(X_test)

        logging.info("SUCCESS: Data has been splitted successfully ")
    except KeyError as err:
        logging.error('FAILED: Data wasn\'t splitted')
        raise err


def test_train_models(train_models):
    '''
    Test train_models
    '''
    # Load the DataFrame
    dataframe = ch.import_data("./data/bank_data.csv")

    # Churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val=="Existing Customer" else 1)

    # Feature engineering 
    (X_train, X_test, y_train, y_test) = ch.perform_feature_engineering(  
                                                    dataframe,
                                                    'Churn')

    # Assert if logistic model is saved
    try:
        ch.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('SUCCESS: Logistic Model was saved')
    except AssertionError as err:
        logging.error('FAILED: Logistic Model wasn\'t saved')
        raise err

    # Assert if Random Forest model is saved
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info("SUCCESS: Random Forest Model was saved")
    except AssertionError as err:
        logging.error('FAILED: Random Forest Model wasn\'t saved')
        raise err

    # Assert if roc curve is saved
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('SUCCESS: roc curve is saved')
    except AssertionError as err:
        logging.error('FAILED: roc curve wasn\' saved')
        raise err

    # Assert if feature importances is saved
    try:
        assert os.path.isfile('./images/results/feature_importances.png') is True
        logging.info('SUCCESS: feature importances is saved')
    except AssertionError as err:
        logging.error('FAILED: feature importances wasn\'t saved')
        raise err


if __name__ == "__main__":

    # Test Import DataFrame
    test_import(ch.import_data)

    # Test Encoding Churn
    test_eda(ch.perform_eda)

    # Test Adding categorical columns
    test_encoder_helper(ch.encoder_helper)

    # Test Freature engineering
    test_perform_feature_engineering(ch.perform_feature_engineering)

    # Test Train models
    test_train_models(ch.train_models)