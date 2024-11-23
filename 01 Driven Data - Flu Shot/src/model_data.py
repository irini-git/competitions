import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

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
                ('imputer', SimpleImputer(strategy='median', missing_values=np.nan)),
                ('scaler', StandardScaler())
                ]
            )

        # Make our ColumnTransformer. (preprocessor)
        numeric_features = self.numeric_features
        categorical_features_binary = ['sex', 'rent_or_own']
        categorical_features = ['employment_occupation', 'employment_industry', 'employment_status', 'census_msa', 'education', 'age_group', 'hhs_geo_region']

        # testing
        # print(numeric_features)
        # X_toy = X['behavioral_wash_hands'].to_frame().reset_index()
        # X_toy = X['behavioral_wash_hands'].copy()
        # print(X_toy)
        # X_toy_ohe = numeric_transformer.fit_transform(X_toy)
        # print(X_toy_ohe)


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

        # Make
        main_pipe = Pipeline(
            steps=[
                ("preprocessor", col_transformer),  # <-- this is the ColumnTransformer we just created
                ("classifier", KNeighborsRegressor())])


        y_h1n1 = y.iloc[:, 1]
        y_seasonal = y.iloc[:, 2]

        for y in [y_h1n1, y_seasonal]:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # We can then use cross_validate() and find our mean training and validation scores!
            with_categorical_scores = cross_validate(main_pipe, X_train, y_train, return_train_score=True)
            categorical_score = pd.DataFrame(with_categorical_scores)
            print(categorical_score)
            print('-'*30)

        # main_pipe.fit(np.array(X_train), np.array(y_train))
        # print(main_pipe.score(X_train, y_train))

        #for y in [y_h1n1]:
           # main_pipe = Pipeline(
            #    steps=[
            #        ("preprocessor", col_transformer),  # <-- this is the ColumnTransformer we just created
            #        ("classifier", MLkNN(k=5))])

            # main_pipe.fit(X, y)
            # print(main_pipe.best_params_, main_pipe.best_score_)

        #     print("model score: %.3f" % clf.score(X_test, y_test))

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

        # 3.2 Doctor recommendations and related to health ready for a pipeline
        # no manual transformation
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