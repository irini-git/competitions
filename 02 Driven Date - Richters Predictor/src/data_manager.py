import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
FILENAME_TEST_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Test_Values.csv'
FILENAME_TRAIN_VALUES = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Labels.csv'
FILENAME_TRAIN_LABELS = '../data/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_Values.csv'
CUTOFF_COLUMNS_DROP = 0.001

class EarthquakeData:
    def __init__(self):
        self.df_train_labels, self.df_train_values, self.df_test_values, self.df_train = self.load_data()
        # self.plot_data()
        self.df_train_cleaned = self.clean_binary_features()
        self.explore_data()


    def explore_data(self):

        def calculate_proportions(c):
                temp = self.df_train_cleaned[c].value_counts()
                temp = temp.to_frame().reset_index()

                # Add a column with feature name
                temp['feature'] = temp.columns[0]

                # Rename columns
                temp.rename(columns={c: 'value'}, inplace=True)

                return temp

        # Prepare data
        columns_explore = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
        df = pd.DataFrame(None, columns=['value', 'count', 'feature'])

        for c in columns_explore:
            temp = calculate_proportions(c)
            df = pd.concat([temp, df])

        level3 = 'geo_level_3_id'
        level2 = 'geo_level_2_id'
        level1 = 'geo_level_1_id'

        # Support scatter plot for geo_level_3_id
        chart_3 = alt.Chart(df.query('feature==@level3')).mark_circle(size=60).encode(
            x='value',
            y='count',
            color='feature',
            # tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
        ).interactive()

        # Support scatter plot for geo_level_2_id
        chart_2 = alt.Chart(df.query('feature==@level2')).mark_circle(size=60).encode(
            x='value',
            y='count',
            color='feature',
            # tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
        ).interactive()

        # Support scatter plot for geo_level_1_id
        chart_1 = alt.Chart(df.query('feature==@level1')).mark_circle(size=60).encode(
            x='value',
            y='count',
            color='feature',
            # tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
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


    def clean_binary_features(self):
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

        return df_train_cleaned

    def plot_data(self):
        """
        Plot data
        """

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

        # Plot -------------
        # Plot counts for categorical features
        # for f in ['land_surface_condition', 'foundation_type', 'roof_type', 'plan_configuration',
        #          'ground_floor_type', 'other_floor_type', 'position', 'legal_ownership_status']:

            # Plot counts for obj variables
            # plot_seaborn_counts(feature_=f)

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

