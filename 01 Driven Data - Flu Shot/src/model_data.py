import re
from symbol import parameters

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sympy.abc import alpha
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILENAME_CLEANED_DATA_FEATURES = '../data/data_train.pkl'
FILENAME_DATA_TARGET_TRAIN = '../data/target_train.pkl'

# Set a random seed for reproducibility
RANDOM_SEED = 6

class CleanedFluShotData:
    """
    Class responsible for the feature engineering and encoding of Flu Shot Data.
    Raw data is loaded from the original file and not from explore_data
    because the exploration step may be omitted.
    """
    def __init__(self):
        self.df_labels, self.df_features = self.load_data()
        self.categorical_features, self.numeric_features = self.split_categorical_numerical_features()
        data_train = self.feature_engineering(self.df_features)
        # data_test = self.feature_engineering(self.df_test)
        self.create_model(data_train, self.df_labels)

    def load_data(self):
        """
        Load raw data (features and labels) from csv files.
        :return: raw train and test data
        """
        df_labels = pd.read_csv(FILENAME_INPUT_DATA_LABELS)
        df_features = pd.read_csv(FILENAME_INPUT_DATA_FEATURES)

        # Pickle target data
        df_labels.to_pickle(FILENAME_DATA_TARGET_TRAIN)

        return df_labels, df_features

    def prepare_sumbission(self):
        """
        Placeholder to create a submission in the required format
        :return:
        """

    def evaluate_model(self):
        """
        Placeholder for model validation
        :return:
        """

    def create_model(self, X, y):
        """
        Create the preprocessing pipelines for numeric and categorical data
        :return:
        """

        # Build a pipeline for our dataset.
        # Make three preprocessing pipelines;
        # - one for the categorical,
        # - one for the numeric features,
        # - and one for categorical binary.

        # What happens if there are categories in the test data, that are not in the training data?
        # In the OneHotEncoder we can specify handle_unknown="ignore" which will then create a row with all zeros.
        # That means that all categories that are not recognized to the transformer will appear the same for this feature.
        # binary : sex (male, female), rent_or_own (Own, Rent)

        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

        # missing value = np.nan : otherwise get an error
        categorical_transformer_binary = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
                ('encoder', OneHotEncoder(sparse_output=False, dtype='int', drop="if_binary"))
                ]
            )

        numeric_transformer = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median', missing_values=np.nan))
                ]
            )

        # Make our ColumnTransformer. (preprocessor)
        numeric_features = self.numeric_features
        categorical_features_binary = ['sex', 'rent_or_own']
        categorical_features = ['employment_occupation', 'employment_industry', 'employment_status', 'census_msa', 'education', 'age_group', 'hhs_geo_region']

        # Set remainder="passthrough" to keep the columns in our feature table which do not need any preprocessing.
        col_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical", categorical_transformer, categorical_features),
                ("categorical_binary", categorical_transformer_binary, categorical_features_binary)
            ],
            remainder='passthrough'
        )

        # View pipeline
        # k (int) – number of neighbours of each input instance to take into account
        # pipe = make_pipeline(col_transformer,  MLkNN(k=20))
        # print(pipe)

        # Classifier
        # classifier = MultiOutputClassifier(XGBClassifier())

        estimators = MultiOutputClassifier(
            estimator=LogisticRegression(penalty="l2", C=1)
        )

        # Make pipeline
        main_pipe = Pipeline(
            steps=[
                ("preprocessor", col_transformer),  # <-- this is the ColumnTransformer we created
                ("model", estimators)])

        # Access the parameter keys of the individual estimators
        # model_parameters = [p for p in main_pipe.get_params().keys() if re.search(r'^model__estimator__', p)]
        # print(model_parameters)

        # Change the type of y: to numpy and remove id
        y_ = y.iloc[:,1:].to_numpy()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.33,
                                            shuffle=True,
                                            stratify=y_,
                                            random_state=RANDOM_SEED
                                        )

        # Train model
        main_pipe.fit(X_train, y_train)

        # Predict on evaluation set
        # This competition wants probabilities, not labels
        yhat = main_pipe.predict(X_test)

        preds = main_pipe.predict_proba(X_test)

        # print(preds)
        # print(yhat)

        # Print results
        # From Data Driven
        # The first array is for h1n1_vaccine, and the second array is for seasonal_vaccine.
        print("test_probas[0].shape", preds[0].shape)
        print("test_probas[1].shape", preds[1].shape)

        # From Data Driven
        # The two columns for each array are probabilities for class 0 and class 1 respectively.
        # That means we want the second column (index 1) for each of the two arrays.
        # Let's grab those and put them in a data frame.

        y_preds = pd.DataFrame(
            {
                "h1n1_vaccine": preds[0][:, 1],
                "seasonal_vaccine": preds[1][:, 1],
            },
            index=X_test.index
        )
        print("y_preds.shape:", y_preds.shape)


        # Check the area under the ROC with the roc_auc_score function
        # auc_y1 = roc_auc_score(y_test[:, 0], yhat[:, 0])
        # auc_y2 = roc_auc_score(y_test[:, 1], yhat[:, 1])
        # print("ROC AUC y1: %.4f, y2: %.4f" % (auc_y1, auc_y2))

        # Check the confusion matrices
        # cm_y1 = confusion_matrix(y_test[:, 0], yhat[:, 0])
        # cm_y2 = confusion_matrix(y_test[:, 1], yhat[:, 1])

        # print(f"Confusion matrix for y1 {cm_y1}")
        # print(f"Confusion matrix for y2 {cm_y2}")

        # Check the classification report with the classification_report function
        # cr_y1 = classification_report(y_test[:, 0], yhat[:, 0], zero_division=0)
        # cr_y2 = classification_report(y_test[:, 1], yhat[:, 1], zero_division=0)
        # print(f"Classification report for y1 {cr_y1}")
        # print(f"Classification report for y2 {cr_y2}")

    def split_categorical_numerical_features(self):
        """
        Split categorical and numeric features based on insights from the data exploration
        :return: list of categorical vs. numerical features
        """

        # There are two ways to split categorical and numerical features
        # a. (applied within current approach) manually inspect the content
        # b. (potential to be used) use of make_column_selector.
        # Option b was not chosen here, as we wanted to carefully inspect all features.

        # 1. Features not to be used
        # income_poverty - large N of missing values
        # health_insurance - large N of missing values
        # race - a majority of respondents having the same reply
        # potentially: marital_status due to mostly equal distribution
        # household_children - replaced by if any child in a household
        # h1n1_knowledge - replaced by a feature with the same scale as other ratings
        # h1n1_concern - replaced by a feature with the same scale as other ratings

        features_to_drop = ['income_poverty', 'health_insurance', 'race', 'household_children',
                            'h1n1_knowledge', 'h1n1_concern']

        # 2. Categorical features
        # No manual transformation, ready for the pipeline
        features_employment = ['employment_occupation', 'employment_industry', 'employment_status']
        features_other = ['census_msa', 'education', 'age_group', 'hhs_geo_region', 'sex', 'rent_or_own']
        categorical_features = features_employment + features_other

        # 3. Numerical features
        # 3.1 Binary Yes/No features are recognized as numeric
        # Behavioral features are ready for a pipeline, no manual transformation
        features_behavioral = ['behavioral_wash_hands', 'behavioral_avoidance', 'behavioral_touch_face',
                               'behavioral_large_gatherings', 'behavioral_outside_home',
                               'behavioral_face_mask', 'behavioral_antiviral_meds']

        # 3.2 Doctor recommendations
        features_doctor_recommendations = ['doctor_recc_seasonal', 'doctor_recc_h1n1']
        features_health = ['chronic_med_condition', 'health_worker', 'child_under_6_months']

        # Ready for a pipeline
        features_household = ['household_adults', 'household_child']

        # 3.4 Sentiment features (ratings)
        # Unify the scale so that concern and knowledge aligned with effective, risk and sick from vaccine
        features_sentiment = ['h1n1_conc', 'h1n1_knwl', 'opinion_h1n1_vacc_effective',
                              'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                              'opinion_seas_vacc_effective',
                              'opinion_seas_risk', 'opinion_seas_sick_from_vacc']

        numeric_features = (features_behavioral + features_doctor_recommendations +
                            features_household + features_sentiment)

        return categorical_features, numeric_features

    def feature_engineering(self, df):
        """
        Create new features based on data exploration
        :param df: input dataframe, train or test
        :return: train (or test) data ready for the model
        """

        # Create a numeric feature based on household_children
        # Logic: any child in a household: 1 - Yes, 0 - No
        df['household_child'] = df['household_children'].map(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 1
            }
        )

        # Unify ratings
        df['h1n1_knwl'] = df['h1n1_knowledge'].map({
                    0.0: 1,
                    1.0: 2,
                    2.0: 5
                    }
                )

        df['h1n1_conc'] = df['h1n1_concern'].map({
                    0.0: 1,
                    1.0: 2,
                    2.0: 4,
                    3.0: 5
                    }
                )

        # Q for respondent id : keep or drop
        data = df[self.numeric_features + self.categorical_features].copy()

        return data