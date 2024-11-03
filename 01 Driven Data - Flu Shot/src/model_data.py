import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILENAME_CLEANED_DATA_FEATURES = '../data/data_train.pkl'
FILENAME_DATA_TARGET_TRAIN = '../data/target_train.pkl'

class CleanedFluShotData:
    """
    Class responsible for the feature engineering and encoding of Flu Shot Data.
    Raw data is loaded from the original file and not from explore_data
    because the exploration step may be omitted.
    """
    def __init__(self):
        self.df_labels, self.df_features = self.load_data()
        categorical_features, numeric_features, data_train = self.feature_engineering()
        self.create_model(categorical_features, numeric_features)

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

    def create_model(self, categorical_features, numeric_features):
        """
        Create the preprocessing pipelines for numeric and categorical data
        :return:
        """

        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median', missing_values=np.NaN)),
                ('scaler', StandardScaler())
                ]
            )


        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Append classifier to preprocessing pipeline
        # work in progress, classifier to be changed
        # assign X, y
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # clf.fit(X_train, y_train)
        # print("model score: %.3f" % clf.score(X_test, y_test))


    def feature_engineering(self):
        """
        Creates new features based on existing, use insights from data exploration
        :return:  new features
        """

        # We have identified two ways to split categorical and numerical features
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
        features_categorical = features_employment + features_other

        # 3. Numerical features
        # 3.1 Binary Yes/No features are recognized as numeric
        # Behavioral features are ready for a pipeline, no manual transformation
        features_behavioral = ['behavioral_wash_hands', 'behavioral_avoidance', 'behavioral_touch_face',
                               'behavioral_large_gatherings', 'behavioral_outside_home',
                               'behavioral_face_mask', 'behavioral_antiviral_meds']

        # 3.2 Doctor recommendations and related to health ready for a pipeline
        # no manual transformation
        features_doctor_recommendations = ['doctor_recc_seasonal', 'doctor_recc_h1n1']
        features_health = ['chronic_med_condition', 'health_worker', 'child_under_6_months']

        # 3.3 Household features (adults and children)
        # Create a numeric feature based on household_children
        # Logic: any child in a household: 1 - Yes, 0 - No
        self.df_features['household_child'] = self.df_features['household_children'].map(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 1
            }
        )

        # Ready for a pipeline
        features_household = ['household_adults', 'household_child']

        # 3.4 Sentiment features (ratings)
        # Unify the scale so that concern and knowledge aligned with effective, risk and sick from vaccine
        features_sentiment = ['h1n1_conc', 'h1n1_knwl', 'opinion_h1n1_vacc_effective',
                              'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                              'opinion_seas_vacc_effective',
                              'opinion_seas_risk', 'opinion_seas_sick_from_vacc']

        self.df_features['h1n1_knwl'] = self.df_features['h1n1_knowledge'].map({
                    0.0: 1,
                    1.0: 2,
                    2.0: 5
                    }
                )

        self.df_features['h1n1_conc'] = self.df_features['h1n1_concern'].map({
                    0.0: 1,
                    1.0: 2,
                    2.0: 4,
                    3.0: 5
                    }
                )


        features_numerical = (features_behavioral + features_doctor_recommendations +
                              features_household + features_sentiment)

        data_train = self.df_features[['respondent_id'] + features_numerical + features_categorical].copy()

        # Pickle cleaned data
        data_train.to_pickle(FILENAME_CLEANED_DATA_FEATURES)

        return features_categorical, features_numerical, data_train