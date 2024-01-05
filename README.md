# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- This repository implements a Python package for a machine learning project focusing on predicting credit card churn. The package follows coding best practices (PEP8), adheres to software engineering principles (modular, documented, and tested), and offers flexibility for interactive or command-line execution summeraized in the following:

* Machine learning model development and training for churn prediction.

* Building a modular, documented, and tested Python package.

* Implementing best practices for Python coding, testing, and logging.

* Understanding a real-world data science challenge: customer churn in the financial domain.

## Files and data description
The project is organized with the following directory architecture:
- Folders
    - Data      
        - eda       --> contains output of the data exploration
        - results   --> contains the dataset in csv format
    - images        --> contains model scores, confusion matrix, ROC curve
    - models        --> contains saved models in .pkl format
    - logs          --> log generated druing testing of library.py file

- project files 
    - churn_library.py
    - churn_notebook.ipnyb
    - requirements.txt

- pytest files (unit test file and configuration files)
    - test_churn_script_logging_and_tests.py  

## Running Files
* First install requirments using this command: python -m pip install -r requirements_py3.6.txt

* Second Run the Machine learning pipline using this command python churn_library.py

* Third to Get a linting score you can run pylint filename.py

* Fourth you can test and log the pipline using  python churn_script_logging_and_tests.py

* Finally you can format file on pep8 autopep8 --in-place --aggressive --aggressive filename.py




