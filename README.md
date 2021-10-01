# Configurations
* Change HOME_PATH in CONFIG.py as the current path 

# Data Prepare

## CENSINCOME
* Download data
* Put census-income.data and census-income.test in Data/Downloads/census-income
* Run DataProcess/CENSINCOME/process.py

## COVTYPE
* Download data
* Put covtype.data in Data/Downloads/COVTYPE/
* Run DataProcess/COVTYPE/process.py

## IJCAI18X
* Download data
* Generate train.csv and test.csv with xgboost1.py
* Put train.csv and test.cvs in Data/RawData/IJCAI18X
* Run DataProcess/IJCAI18X/process.py

## KDDCUP19
* Download data
* Put train_queries.csv, train_plans.csv, train_clicks.csv, profiles.csv in Data/Downloads/KDDCUP19P1
* Run DataProcess/KDC/process.py

# Training

## Train and save the model
* Use the scripts in Train/{DATASET}/
* Will train the model and save the model in Data/Saved/{DATASET}/[VOTERS or DNN]/model

## t-test
* Run multiple times of the performance evaluation on each model
* Save the results as .csv files in Ttest/data/{DATASET}/{MODEL_NAME}.csv. One test for each row. The column names are prc and auc.
* Run Ttest/ttest.py

# Explanation

# Generate intermediate results of voting analysis
* Run SaveInter/save.py 

# Visualizations for explanation
* Run E1 to E6 in Explain/analysis, the visualizations will be saved in Explain/out

# Demo system for local/global decision path visualization
* Open the path of the django project in Site/
* Start the server: python manage.py runserver 0.0.0.0:8000
* Use the system in browser with URL: 127.0.0.1:8000/{global/local}/{covtyp/kdc/ijcai/cens}

