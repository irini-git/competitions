# from unicodedata import category

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
import warnings
import pickle

from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import f1_score

# Constants
FILENAME_TEST_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv'
FILENAME_TRAIN_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv'
FILENAME_TRAIN_LABELS = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv'
CUTOFF_COLUMNS_DROP = 0.001
CUTOFF_AREA_PRCT = 35
CUTOFF_HEIGHT_PRCT = 14
PALETTE = ["#FFDE91", "#FE7E03", "#9B1D1E"]
RANDOM_SEED = 6 # Set a random seed for reproducibility

class EarthquakeData:
    def __init__(self):
        self.df_train_labels, self.df_train_values, self.df_test_values, self.df_train = self.load_data()
        self.df_train_cleaned = self.df_train
        self.df_test_values_cleaned = self.df_test_values
        # self.plot_data()
        # self.explore_geo_levels()
        # self.explore_other()
        self.clean_features()
        self.create_model()
        self.explore_feature_importance()

    def explore_feature_importance(self):
        # Plot feature importance
        feature_importances = np.load('../data/feature_importances.npy', allow_pickle=False)

        with open('../data/feature_names.pkl', 'rb') as fp:
            feature_names = pickle.load(fp)

        feature_names = np.array(feature_names)

        # Plot the feature importances of the forest
        sorted_idx = np.argsort(feature_importances)
        fig = plt.figure(figsize=(12,4))
        plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
        plt.tight_layout()
        plt.title('Feature Importance')
        plt.savefig('../fig/Feature_importance016.png')

    def create_model(self):
        """
        Classifier for the problem
        :return: creates model and predictions
        """

        # Define numeric and categorical features -------------
        numeric_features_binary = ['has_superstructure_mud_mortar_stone',
                            'has_superstructure_timber', 'ground_floor_type_f',
                            'foundation_type_cleaned',
                            'land_roof_type_x']

        numeric_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
                            'area_pct_cleaned', 'height_pct_cleaned']

        # Initially, there were some categorical features
        # They were transformed to binary or removed
        categorical_features = []

        # Define pipelines for numeric and categorical features -------------
        numeric_transformer = Pipeline(
            steps=[
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

        ohe_transformer = Pipeline(
            steps=[
                ('encoder', OneHotEncoder(drop="if_binary", handle_unknown='ignore')),
                ]
            )

        # Make our ColumnTransformer
        # Set remainder="passthrough" to keep the columns in our feature table which do not need any preprocessing.
        col_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("numeric_binary", ohe_transformer, numeric_features_binary),
                ("ohe", ohe_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Define classifier
        classifier = RandomForestClassifier(random_state=2018,
                                            # n_estimators=300,
                                            min_samples_leaf=5,
                                            min_samples_split=2,
                                            verbose=4)

        # RandomForestClassifier(random_state=2018)
        # f1 0.7064151908743125 min_samples_leaf 5  n_estimators 300 - ground floor
        # f1 0.7112059442551658 min_samples_leaf 5  n_estimators 300 - ownership
        # f1 0.7116478098582542 min_samples_leaf 5  n_estimators 300 - position
        # f1 0.7127175897394156 min_samples_leaf 5  n_estimators 300 - foundation_type
        # f1 0.713764113536204 min_samples_leaf 5  n_estimators 300 dropped number of floors
        # f1 0.7158804172141536 min_samples_leaf 5  n_estimators 300 - land_surface_condition_t
        # f1 0.7163339108594285 -- removed plan config
        # f1 0.7179036965546111 -- removed position cleaned
        # f1 0.7181013732717823 -- removed ownership status
        # f1 0.7197990674310166 -- removed foundation type
        # f1 0.7206014023418877 -- removed other_floor_type
        # f1 0.72368283352132 -- added land_roof_type_x
        # f1 0.7249502901196525 -- removed secondary use
        # f1 0.7261014662961197 -- removed 'land_surface_condition_t'
        # f1 0.7258223932836428 -- removed 'has_superstructure_timber' <-- did not work
        # --------------------- parameter tuning
        # f1 0.7261014662961197
        # f1 0.7261014662961197
        # f1 0.7255084361446064

        # Make a pipeline
        main_pipe = Pipeline(
            steps=[
                ("preprocessor", col_transformer),  # <-- this is the ColumnTransformer we created
                ("model", classifier)])

        param_grid = {
                     'model__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
                      }

        gs = GridSearchCV(main_pipe, param_grid, cv=2, verbose=4)

        # Define X and y
        # X : contain columns in scope
        # y : contain damage
        y = self.df_train_cleaned['damage_grade'].values
        X = self.df_train_cleaned[numeric_features + numeric_features_binary + categorical_features].copy()

        # Submissions
        test_values_subset = self.df_test_values_cleaned[numeric_features + numeric_features_binary + categorical_features].copy()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                            shuffle=True,
                                            stratify=y,
                                            random_state=RANDOM_SEED
                                        )

        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)
        # For this task, we use f1 micro
        print(f1_score(y_test, y_pred, average="micro"))
        print(gs.best_params_)
        
        # Make predictions
        predictions = gs.predict(test_values_subset)

        # Save data
        np.save('../data/predictions.npy', predictions)

        # Feature exploration ---------------
        feature_importances = gs.best_estimator_._final_estimator.feature_importances_
        np.save('../data/feature_importances.npy', feature_importances)

        # Get the feature names array and save to file
        feature_names = gs.best_estimator_[:-1].get_feature_names_out()
        with open('../data/feature_names.pkl', 'wb') as fp:
            pickle.dump(list(feature_names), fp)

        # Correlation
        # Uncomment if you want to plot correlation
        # corrMatrix = X_train[numeric_features_binary + numeric_features].corr()
        # sns.heatmap(corrMatrix, annot=True)
        # plt.show()
        # plt.figure(figsize=(15, 8))
        # plt.savefig('../fig/Correlation.png')

    def create_submission(self):

        # Load data from files
        predictions = np.load('../data/predictions.npy')
        submission_id = self.df_test_values_cleaned['building_id'].values

        my_submission = pd.DataFrame(
                {'building_id': submission_id[:], 'damage_grade': predictions[:]})
        
        # Create submission
        my_submission.to_csv('../data/mysubmission_RandomForestClassifiercsv_003.csv', index=False)


    def clean_features(self):
        """
        1. Before modelling :

        - area_percentage (type: int):
        normalized area of the building footprint.
        Introduce 'larger than CUTOFF_AREA_PRCT'

        - height_percentage (type: int):
        normalized height of the building footprint.
        Introduce 'larger than CUTOFF_HEIGHT_PRCT'

        - count_floors_pre_eq : transform to binary
        less than 4 or larger than four

        - age of building n years :
        not performed ---------
        there are rows where value is 995,
        replace with median
            value damage_grade  count feature
           995            3    389     age
           995            2    822     age
           995            1    179     age

        2. After modeling, based on feature importance
        all that are not in the list
        :return: transformed features
        """

        # Support functions for cleaned features
        def create_cleaned_area(row):
            if row['area_percentage'] < CUTOFF_AREA_PRCT:
                return row['area_percentage']
            else:
                return CUTOFF_AREA_PRCT

        # Based on feature importance, not used
        def create_cleaned_floor(row):
            if row['count_floors_pre_eq'] < 4: # 0.5689135920185119
                return 1
            else:
                return 0

        def create_cleaned_height(row):
            if row['height_percentage'] < CUTOFF_HEIGHT_PRCT:
                return row['height_percentage']
            else:
                return CUTOFF_HEIGHT_PRCT

        # Plan configuration to binary: if d or not
        # Will be not used based on feature importance
        def create_cleaned_plan(row):
            if row['plan_configuration'] == 'd':
                return 1
            else:
                return 0

        # Group ground_floor_type f or not f
        def create_cleaned_ground_floor(row):
            if row['ground_floor_type'] == 'f':
                return 1
            else:
                return 0

        # Group legal_ownership_status f or not f
        # Based on feature importance, will not use
        def create_cleaned_legal_ownership_status(row):
            if row['legal_ownership_status'] == 'v':
                return 1
            else:
                return 0

        # Group positition : combine o and j to one group, keep s and t as is
        # Based on feature importance, will not use
        def create_cleaned_position(row):
            if row['position'] in ['s', 't']:
                return row['position']
            else:
                # o for other
                return 'o'

        # Group foundation_type : r or other
        def create_cleaned_foundation_type(row):
            if row['foundation_type'] == 'r':
                return 1
            else:
                # o for other
                return 0

        # Land surface condition : t or not t
        def create_cleaned_land_surface(row):
            if row['land_surface_condition'] == 't':
                return 1
            else:
                return 0

        # Roof type : x or not
        # Based on feature importance, x has more importance even if smaller
        def create_cleaned_roof_type(row):
            if row['roof_type'] == 'x':
                return 1
            else:
                return 0

        # Replace age of building equal to 995 years by the median of the column
        for df in [self.df_train_cleaned, self.df_test_values_cleaned]:
            #median_age = int(np.median(df['age']))
            # df['age'] = df['age'].replace({995: median_age})

            # Roof type:  keep x or not
            # Used
            df['land_roof_type_x'] = df.apply(create_cleaned_roof_type, axis=1)

            # Land surface condition :  keep s and t
            # Not used based on feature importance
            # df['land_surface_condition_t'] = df.apply(create_cleaned_land_surface, axis=1)

            # foundation_type :  keep s and t
            # Used
            df['foundation_type_cleaned'] = df.apply(create_cleaned_foundation_type, axis=1)

            # position :  keep s and t
            # Not used based on feature importance
            # df['position_cleaned'] = df.apply(create_cleaned_position, axis=1)

            # legal_ownership_status_v
            # Not used based on feature importance
            # df['legal_ownership_status_v'] = df.apply(create_cleaned_legal_ownership_status, axis=1)

            # ground_floor_type f
            df['ground_floor_type_f'] = df.apply(create_cleaned_ground_floor, axis=1)

            # floor
            # Not used based on feature importance
            # df['n_floors_cleaned'] = df.apply(create_cleaned_floor, axis=1)

            # area_percentage
            # Used
            df['area_pct_cleaned'] = df.apply(create_cleaned_area, axis=1)

            # height_percentage
            # Used
            df['height_pct_cleaned'] = df.apply(create_cleaned_height, axis=1)

            # plan_configuration
            # Not used based on feature importance
            # df['plan_config_cleaned'] = df.apply(create_cleaned_plan, axis=1)


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
   

    def plot_data(self):
        """
        Plot data
        """

        def plot_seaborn_counts(feature_):

            plt.figure(figsize=(10,6))
            sns.countplot(data=self.df_train,
                          x = feature_,
                          hue='damage_grade',
                          palette=sns.color_palette(PALETTE, len(PALETTE)))

            sns.set_theme(font_scale=1.4)

            plt.title(f'Damage vs {feature_}')
            plt.xlabel(f'{feature_}')
            plt.ylabel('')

            plt.savefig(f'../fig/Explore_count_{feature_}.png')


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
