import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np

# Constants
FILENAME_TEST_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv'
FILENAME_TRAIN_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv'
FILENAME_TRAIN_LABELS = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv'


class EarthquakeData:
    def __init__(self):
        self.df_train_labels, self.df_train_values, self.df_test_values, self.df_train = self.load_data()
        self.explore_data()

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

    def explore_data(self):
        """
        Explore data
        """

        # Explore columns, shape
        print('-'*10)
        for df in [self.df_train]:
            # for c in df.columns:
                # print(df[c].value_counts())
            print(f'Dataframe shape : {df.shape}')
            print(f'Null values in database : {df.isnull().sum().sum()}')
            print(df.info())
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df.describe(include='all'))

        def plot_seaborn_counts(feature_):
            # Set the palette to the "pastel" default palette:
            # sns.set_palette("pastel")

            # plt.style.use("seaborn-v0_8-whitegrid")
            palette = ["#FFDE91", "#FE7E03", "#9B1D1E"]

            plt.figure(figsize=(10,6))
            sns.countplot(data=self.df_train,
                          x = feature_,
                          hue='damage_grade',
                          palette=sns.color_palette(palette, len(palette)))


            sns.set_theme(font_scale=1.4)

            plt.title(f'Damage vs {feature_}')
            plt.xlabel(f'{feature_}')
            plt.ylabel('')

            # plt.rcParams['axes.spines.top'] = False
            # plt.rcParams['axes.spines.right'] = False

            plt.savefig(f'../fig/Explore_count_{feature_}.png')

        # Plot counts for categorical features
        #  land_surface_condition                  260601 non-null  object
        #  foundation_type                         260601 non-null  object
        #  roof_type                               260601 non-null  object
        #  ground_floor_type                       260601 non-null  object
        #  other_floor_type                        260601 non-null  object
        #  position                                260601 non-null  object
        #  plan_configuration                      260601 non-null  object
        #  legal_ownership_status                  260601 non-null  object
        # for f in ['land_surface_condition', 'foundation_type', 'roof_type', 'plan_configuration',
        #          'ground_floor_type', 'other_floor_type', 'position', 'legal_ownership_status']:

            # Plot counts for obj variables
            # plot_seaborn_counts(feature_=f)
            # pass

        def plot_altair_counts(binary_columns, title_):

            def create_support_df(feature_):
                """
                Create support pivoted data for binary features
                Used as input for plots
                :return:
                """

                # print(self.df_train[['has_secondary_use_other', 'damage_grade']].head(10))
                temp = self.df_train.groupby([feature_,  'damage_grade'],as_index=False)['building_id'].count()

                # Add a column with feature name
                temp['feature'] = temp.columns[0]

                # Rename columns
                return temp.rename({'building_id': 'count', feature_: 'value'}, axis='columns')

            # Prepare data
            df = pd.DataFrame(None, columns=['value','damage_grade','count', 'feature'])
            for c in binary_columns:
                temp = create_support_df(c)
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

        # Has secondary columns
        features_has_secondary_use = ['has_secondary_use_other', 'has_secondary_use_use_police',
                          'has_secondary_use_gov_office', 'has_secondary_use_health_post',
                          'has_secondary_use_industry', 'has_secondary_use_school',
                          'has_secondary_use_institution', 'has_secondary_use_rental',
                          'has_secondary_use_hotel', 'has_secondary_use_agriculture',
                          'has_secondary_use']
        title_has_secondary_use = 'Has secondary use'
        plot_altair_counts(features_has_secondary_use, title_has_secondary_use)

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

