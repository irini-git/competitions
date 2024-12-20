import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
import warnings

from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# Constants
FILENAME_TEST_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv'
FILENAME_TRAIN_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv'
FILENAME_TRAIN_LABELS = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv'
CUTOFF_COLUMNS_DROP = 0.001
CUTOFF_AREA_PRCT = 35
CUTOFF_HEIGHT_PRCT = 14
PALETTE = ["#FFDE91", "#FE7E03", "#9B1D1E"]
# Set a random seed for reproducibility
RANDOM_SEED = 6

class EarthquakeData:
    def __init__(self):
        self.df_train_labels, self.df_train_values, self.df_test_values, self.df_train = self.load_data()
        self.df_train_cleaned = self.df_train
        self.df_test_values_cleaned = self.df_test_values
        # self.plot_data()
        # self.clean_features()
        # self.explore_geo_levels()
        # self.explore_other()
        self.clean_numeric_features()
        self.create_model()

    def create_model(self):
        """
        Classifier for the problem
        :return:
        """

        # Define numeric and categorical features -------------
        numeric_features = ['count_floors_pre_eq', 'geo_level_1_id',
                            'area_pct_cleaned', 'height_pct_cleaned',
                            'has_superstructure_mud_mortar_stone',
                            'has_superstructure_timber',
                            'has_secondary_use', 'plan_config_cleaned']
        # 'has_secondary_use_agriculture', 'has_secondary_use_hotel'

        categorical_features = ['land_surface_condition', 'foundation_type',
                                'roof_type', 'ground_floor_type',
                                'other_floor_type', 'position',
                                'legal_ownership_status']

        # Define pipelines for numeric and categorical features -------------
        numeric_transformer = Pipeline(
            steps=[
                ('encoder', OneHotEncoder(drop="if_binary",
                                          handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ]
            )

        categorical_transformer = Pipeline(
            steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

        # Make our ColumnTransformer. (preprocessor)
        # Set remainder="passthrough" to keep the columns in our feature table which do not need any preprocessing.
        col_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical", categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Define classifier
        # 'model__min_samples_leaf': [5]
        classifier = RandomForestClassifier(random_state=2018,
                                            min_samples_leaf=5)


        # RandomForestClassifier(random_state=2018)
        # f1 0.6892056884382377
        # f1 0.6877754392492936

        # Make a pipeline
        main_pipe = Pipeline(
            steps=[
                ("preprocessor", col_transformer),  # <-- this is the ColumnTransformer we created
                ("model", classifier)])

        param_grid = {'model__n_estimators': [50, 100]}

        gs = GridSearchCV(main_pipe, param_grid, cv=2, verbose=4)

        # Define X and y
        # X : remove id and damage, and other columns out of scope
        # y : only contain damage
        y = self.df_train_cleaned['damage_grade'].values
        X = self.df_train_cleaned[numeric_features + categorical_features].copy()

        # Submissions
        submission_id = self.df_test_values_cleaned['building_id'].values
        test_values_subset = self.df_test_values_cleaned[numeric_features + categorical_features].copy()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                            shuffle=True,
                                            stratify=y,
                                            random_state=RANDOM_SEED
                                        )
        gs.fit(X_train, y_train)
        print(gs.best_params_)
        y_pred = gs.predict(X_test)
        print(f1_score(y_test, y_pred, average="micro"))
        predictions = gs.predict(test_values_subset)

        my_submission = pd.DataFrame(data=predictions,
                                     columns=['building_id','damage_grade'],
                                     index=submission_id)

        my_submission.to_csv('../data/submission_001.csv')


    def clean_numeric_features(self):
        """
        There are two numeric features that we will clean

        - area_percentage (type: int):
        normalized area of the building footprint.
        Introduce 'larger than CUTOFF_AREA_PRCT'

        - height_percentage (type: int):
        normalized height of the building footprint.
        Introduce 'larger than CUTOFF_HEIGHT_PRCT'

        - age of building n years :
        there are rows where value is 995,
        replace with median
            value damage_grade  count feature
           995            3    389     age
           995            2    822     age
           995            1    179     age
        :return:
        """

        # Support functions for cleaned features
        def create_cleaned_area(row):
            if row['area_percentage'] < CUTOFF_AREA_PRCT:
                return row['area_percentage']
            else:
                return CUTOFF_AREA_PRCT

        def create_cleaned_height(row):
            if row['height_percentage'] < CUTOFF_HEIGHT_PRCT:
                return row['height_percentage']
            else:
                return CUTOFF_HEIGHT_PRCT

        # Plan configuration to binary: if d or not
        def create_cleaned_plan(row):
            if row['plan_configuration'] == 'd':
                return 1
            else:
                return 0

        # From exploration, some 'has secondary use'
        # to be dropped -
        # rational : almost all values are 0 (does not have secondary use)
        # and similar distrbution btw damage grade

        columns_to_drop_secondary_use = ['has_secondary_use_gov_office', 'has_secondary_use_health_post',
                                        'has_secondary_use_industry', 'has_secondary_use_institution',
                                        'has_secondary_use_other', 'has_secondary_use_rental',
                                        'has_secondary_use_school', 'has_secondary_use_use_police']

        # Replace age of building equal to 995 years by the median of the column
        for df in [self.df_train_cleaned, self.df_test_values_cleaned]:
            median_age = int(np.median(df['age']))
            df['age'] = df['age'].replace({995: median_age})

            # area_percentage
            df['area_pct_cleaned'] = df.apply(create_cleaned_area, axis=1)

            # height_percentage
            df['height_pct_cleaned'] = df.apply(create_cleaned_height, axis=1)

            # plan_configuration
            df['plan_config_cleaned'] = df.apply(create_cleaned_plan, axis=1)

            # Drop initial area, height and age features
            df.drop(['area_percentage', 'height_percentage', 'age', 'plan_configuration'], axis='columns', inplace=True)

            # Drop some secondary use
            df.drop(columns_to_drop_secondary_use, axis='columns', inplace=True)

    def explore_other(self):
        """
        Explore four other numeric features
        :return: plots
        """
        columns_explore = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage']
        df = pd.DataFrame(None, columns=['value', 'damage_grade', 'count', 'feature'])

        # create_support_df(self, df, feature_)
        for c in columns_explore:
            temp = self.create_support_df(self.df_train_cleaned, c)
            df = pd.concat([temp, df])

        feature1 = 'count_floors_pre_eq'
        feature2 ='age'
        feature3 = 'area_percentage'
        feature4 = 'height_percentage'

        # Scatter plot for floor number
        chart1 = alt.Chart(df.query('feature==@feature1'),
                           title='Number of floors in the building before the earthquake').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            color=alt.Color('damage_grade').scale(scheme='yelloworangebrown')
        ).interactive()

        chart2 = alt.Chart(df.query('feature==@feature2'),
                           title='Age of the building in years').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            color=alt.Color('damage_grade')
        ).interactive()

        chart3 = alt.Chart(df.query('feature==@feature3'),
                           title='Normalized area of building').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            color=alt.Color('damage_grade')
        ).interactive()

        chart4 = alt.Chart(df.query('feature==@feature4'),
                           title='Normalized height of building').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            color=alt.Color('damage_grade')
        ).interactive()

        upper = chart1 | chart2
        lower = chart3 | chart4

        chart = alt.vconcat(upper, lower)

        chart.save(f'../fig/Explore_altair_numeric_count_floors_age_area_height.png')

    def explore_geo_levels(self):

        # Prepare data
        columns_explore = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
        df = pd.DataFrame(None, columns=['value', 'damage_grade', 'count', 'feature'])

        for c in columns_explore:
            temp = self.create_support_df(self.df_train, c)
            df = pd.concat([temp, df])

        level3 = 'geo_level_3_id'
        level2 = 'geo_level_2_id'
        level1 = 'geo_level_1_id'

        # Support scatter plot for geo_level_3_id
        chart_3 = alt.Chart(df.query('feature==@level3'), title='Geographic region 3').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            # color=alt.Color('damage_grade', legend=None)
            color = alt.Color('damage_grade').scale(scheme='yelloworangebrown')
        ).interactive()

        # Support scatter plot for geo_level_2_id
        chart_2 = alt.Chart(df.query('feature==@level2'), title='Geographic region 2').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            color=alt.Color('damage_grade')
        ).interactive()

        # Support scatter plot for geo_level_1_id
        chart_1 = alt.Chart(df.query('feature==@level1'), title='Geographic region 1').mark_circle(size=60).encode(
            x=alt.X('value', title=''),
            y=alt.Y('count', title=''),
            color=alt.Color('damage_grade')
        ).interactive()

        chart = chart_1 | chart_2 | chart_3

        chart.save(f'../fig/Explore_altair_numeric_geo_level.png')

    def load_data(self):
        """
        Load train and test data
        :return:
        """

        df_train_labels = pd.read_csv(FILENAME_TRAIN_LABELS)
        df_train_values = pd.read_csv(FILENAME_TRAIN_VALUES)
        df_test_values = pd.read_csv(FILENAME_TEST_VALUES)

        df_train = df_train_labels.merge(df_train_values)

        return df_train_labels, df_train_values, df_test_values, df_train


    def clean_features(self):
        """
        Define features to drop
        :return:
        """

        columns_explore = [
       'land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position', 'legal_ownership_status',
       'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
       'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel',
       'has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry',
       'has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other']

        def calculate_proportions(c):
                temp = self.df_train_labels[c].value_counts(normalize=True)
                temp = temp.to_frame().reset_index()

                # Add a column with feature name
                temp['feature'] = temp.columns[0]

                # Rename columns
                temp.rename(columns={c: 'value'}, inplace=True)

                return temp

        # Prepare data
        df = pd.DataFrame(None, columns=['value', 'proportion', 'feature'])
        for c in columns_explore:
            temp = calculate_proportions(c)
            df = pd.concat([temp, df])

        df = df.sort_values(by='proportion', ascending=True)
        columns_to_drop = df.query('proportion<@CUTOFF_COLUMNS_DROP')['feature'].values
        values_to_drop = df.query('proportion<@CUTOFF_COLUMNS_DROP')['value'].values
        features_drop_dict = {k: v for k, v in zip(columns_to_drop, values_to_drop)}

        # Drop values less than cut-off in specific columns
        df_train_cleaned = pd.DataFrame()
        for k,v in features_drop_dict.items():
            df_train_cleaned = self.df_train[(self.df_train[k] != v)]


        self.df_train_cleaned = df_train_cleaned

    def plot_data(self):
        """
        Plot data
        """

        def plot_seaborn_counts(feature_):
            # Set the palette to the "pastel" default palette:
            # sns.set_palette("pastel")

            plt.figure(figsize=(10,6))
            sns.countplot(data=self.df_train,
                          x = feature_,
                          hue='damage_grade',
                          palette=sns.color_palette(PALETTE, len(PALETTE)))


            sns.set_theme(font_scale=1.4)

            plt.title(f'Damage vs {feature_}')
            plt.xlabel(f'{feature_}')
            plt.ylabel('')

            # plt.rcParams['axes.spines.top'] = False
            # plt.rcParams['axes.spines.right'] = False

            plt.savefig(f'../fig/Explore_count_{feature_}.png')

        # Plot -------------
        # Plot counts for categorical features
        # for f in ['land_surface_condition', 'foundation_type', 'roof_type', 'plan_configuration',
        #          'ground_floor_type', 'other_floor_type', 'position', 'legal_ownership_status']:

            # Plot counts for obj variables
            # plot_seaborn_counts(feature_=f)

        def plot_altair_counts(binary_columns, title_):


            # Prepare data
            df = pd.DataFrame(None, columns=['value','damage_grade','count', 'feature'])
            for c in binary_columns:
                temp = self.create_support_df(self.df_train, c)
                df = pd.concat([temp, df])

            # Plot
            chart = alt.Chart(df).mark_bar().encode(
                column="value:O",
                x="count",
                y="feature",
                color="damage_grade",
            ).properties(
                width=220,
                title=f'{title_}'
            )

            chart.save(f'../fig/Explore_altair_numeric_{title_.lower()}.png')

        # Plot -------------
        # Has secondary columns
        features_has_secondary_use = ['has_secondary_use_other', 'has_secondary_use_use_police',
                          'has_secondary_use_gov_office', 'has_secondary_use_health_post',
                          'has_secondary_use_industry', 'has_secondary_use_school',
                          'has_secondary_use_institution', 'has_secondary_use_rental',
                          'has_secondary_use_hotel', 'has_secondary_use_agriculture',
                          'has_secondary_use']
        title_has_secondary_use = 'Has secondary use'
        plot_altair_counts(features_has_secondary_use, title_has_secondary_use)

        # Plot -------------
        # Has superstructure
        features_has_superstructure = ['has_superstructure_other', 'has_superstructure_rc_engineered',
                                       'has_superstructure_rc_non_engineered',
                                       'has_superstructure_bamboo',
                                       'has_superstructure_timber',
                                       'has_superstructure_cement_mortar_brick',
                                       'has_superstructure_mud_mortar_brick',
                                       'has_superstructure_cement_mortar_stone',
                                       'has_superstructure_stone_flag',
                                       'has_superstructure_mud_mortar_stone',
                                       'has_superstructure_adobe_mud']
        title_has_superstructure = 'Has superstructure'
        plot_altair_counts(features_has_superstructure, title_has_superstructure)

    def create_support_df(self, df, feature_):
        """
         Create support pivoted data for binary features
        Used as input for plots
        :return:
        """

        # print(self.df_train[['has_secondary_use_other', 'damage_grade']].head(10))
        temp = df.groupby([feature_,  'damage_grade'],as_index=False)['building_id'].count()

        # Add a column with feature name
        temp['feature'] = temp.columns[0]

        # Rename columns
        return temp.rename({'building_id': 'count', feature_: 'value'}, axis='columns')
