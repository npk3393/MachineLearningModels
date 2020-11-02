# Disaster Response Pipeline Project

### Motivation:

This project seeks to train a multi class classifier using Random Forest to identify an incoming emergency text, parse it to identify the category (nature of) the emergency.

This data science project uses python flask framework as a front end, leverages on SQLite to create relational database to store the text output, and utilizes scikit learn's Random Forest along with Grid Search to fine tune the hyper parameters for the model.


### Instructions:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
ls

3. Go to http://0.0.0.0:3001/

