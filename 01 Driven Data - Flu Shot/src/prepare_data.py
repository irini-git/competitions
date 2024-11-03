import pandas as pd
from narwhals.selectors import categorical
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

class CleanedFluShotData:
    """
    Class responsible for the feature engineering and encoding of Flu Shot Data.
    Raw data is loaded from the original file and not from explore_data
    because the exploration step may be omitted.
    """
    def __init__(self):
        self.df_labels, self.df_features = self.load_data()
        self.feature_engineering()

    def load_data(self):
        """
        Load raw data (features and labels) from csv files.
        :return: raw train and test data
        """
        df_labels = pd.read_csv(FILENAME_INPUT_DATA_LABELS)
        df_features = pd.read_csv(FILENAME_INPUT_DATA_FEATURES)

        return df_labels, df_features

    def create_model(self):
        """
        Create a model, work in progress
        :return:
        """

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))])

        numerical_transformer = []

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, [0])
            ])

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
        features_to_drop = ['income_poverty', 'health_insurance', 'race', 'household_children']

        # 2. Categorical features
        # No manual transformation, ready for the pipeline
        features_employment = ['employment_occupation', 'employment_industry', 'employment_status']
        features_other = ['census_msa', 'education', 'age_group', 'hhs_geo_region', 'sex', 'rent_or_own']

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

        for f in features_household:
            print(self.df_features[f].value_counts())
            print(f'Null values : {self.df_features[f].isna().sum()}')
            print('-'*10)

        # 3.4 Sentiment features (ratings)
        # Unify the scale
        features_sentiment = ['h1n1_concern', 'h1n1_knowledge', 'opinion_h1n1_vacc_effective',
                              'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                              'opinion_seas_vacc_effective',
                              'opinion_seas_risk', 'opinion_seas_sick_from_vacc']




        # pickle output data