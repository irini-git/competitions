import pandas as pd
import time
import datetime
import logging
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Constants
TEST_DATA_FEATURES = '../data/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv'
TRAINING_DATA_FEATURES = '../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv'
TRAINING_DATA_LABELS = '../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv'

COLORHEX_GREY = '#767676'
COLORHEX_ASCENT = '#ff4d00'


# Timestamp for a log file
ts = time.time()
ts_ = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%Hh%M')
# FILENAME_LOGGING = f'../data/logging_{ts_}.log'
FILENAME_LOGGING = f'../data/log.log'

# Parameters for logging
logging.basicConfig(level=logging.INFO, filename=FILENAME_LOGGING, filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

class DengueData:
    def __init__(self):
        self.train_data, self.test_data_features = self.load_data()


    def load_data(self):
        """
        Load data for the challenge
        :return: raw unprocessed dataframes with train and test data
        """
        # Load data from csv files
        train_data_features = pd.read_csv(TRAINING_DATA_FEATURES)
        train_data_labels = pd.read_csv(TRAINING_DATA_LABELS)
        test_data_features = pd.read_csv(TEST_DATA_FEATURES)

        # Merge features and labels for exploration
        train_data = pd.merge(
            left=train_data_features,
            right=train_data_labels,
            how='inner',
            on=['city',  'year',  'weekofyear']
        )

        # Visual exploration of the first two rows
        logging.info(f"Train data labels\n {train_data_labels.head(2)}\n")
        logging.info(f"Train data features\n {train_data_features.head(2)}\n")
        logging.info(f"Train data combined\n {train_data.head(2)}\n")

        # Column names in plain English
        train_data.rename(columns={"station_diur_temp_rng_c": "Diurnal temperature range station",
                                   "station_precip_mm": "Total precipitation station station",
                                   "station_min_temp_c": "Minimum temperature station",
                                   "station_max_temp_c": "Maximum temperature station",
                                   "station_avg_temp_c": "Average temperature station",
                                   "precipitation_amt_mm": "Total precipitation station satellite",
                                   "reanalysis_sat_precip_amt_mm" : "Total precipitation mm NCEP",
                                   "reanalysis_dew_point_temp_k": "Mean dew point temperature NCEP",
                                   "reanalysis_air_temp_k": "Mean air temperature forecast",
                                   "reanalysis_relative_humidity_percent": "Mean relative humidity NCEP",
                                   "reanalysis_specific_humidity_g_per_kg":"Mean specific humidity NCEP",
                                   "reanalysis_precip_amt_kg_per_m2":"Total precipitation kg_per_m2 NCEP",
                                   "reanalysis_max_air_temp_k": "Maximum air temperature NCEP",
                                   "reanalysis_min_air_temp_k":"Minimum air temperature NCEP",
                                   "reanalysis_avg_temp_k":"Average air temperature NCEP",
                                   "ndvi_se":"Pixel southeast of city centroid",
                                   "ndvi_sw":"Pixel southwest of city centroid",
                                   "ndvi_ne":"Pixel northeast of city centroid",
                                   "ndvi_nw":"Pixel northwest of city centroid",
                                   "reanalysis_tdtr_k": "Diurnal temperature range forecast"
                                   },
                          inplace=True)

        print(train_data.columns)

        return train_data, test_data_features

    def clean_data(self):

        # Explore data
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.train_data.head(2))
            print(self.train_data.info())
            # for c in self.train_data.columns:
            #     print(self.train_data[c].value_counts().head(2))

        # Parse date column to datetime format
        self.train_data['date'] = pd.to_datetime(self.train_data['week_start_date'], format='%Y-%m-%d')

        # Print to screen
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.train_data['week_start_date'].head(3))

        # Check if chronological order
        self.train_data = self.train_data.sort_values(by='date')
        print(f"Is monotonic increasing : {self.train_data['date'].is_monotonic_increasing}")


        # Visualization

        # ----------------
        # Create an array with the colors you want to use
        colors = ["#b3b7bd", "#1457ba"]
        # Set your custom color palette
        sns.set_palette(sns.color_palette(colors))

        # ----------------
        def plot_raw_features(term):
            columns_to_visualize = [c for c in self.train_data if term in c]
            # print(columns_to_visualize)

            f, ax = plt.subplots(nrows=len(columns_to_visualize), ncols=1, figsize=(15, 5*len(columns_to_visualize)))

            for i, column in enumerate(columns_to_visualize):
                sns.lineplot(x=self.train_data['date'],
                             y=self.train_data[column].ffill(),
                             ax=ax[i],
                             hue=self.train_data['city'])
                ax[i].set_title(f'{column}')
                ax[i].set_xlabel('')
                ax[i].set_ylabel('')

                ax[i].set_xlim([datetime.date(1990, 4, 30),
                                datetime.date(2010, 6, 25)])

            f.savefig(f'../fig/Explore_raw_{term}.png')
        # ----------------

        def plot_with_missing(df, figure_name, term):

            # Color grey
            hex_grey_color = '#767676'
            # Ascent color
            ascent_hex_color = '#ff4d00'

            features = [c for c in self.train_data if term in c]

            # Plot raw features with red missing values
            f, ax = plt.subplots(nrows=len(features), ncols=2, figsize=(22, 4*len(features)))

            for i, feature in enumerate(features):

                # San Juan
                old_feature_sj = df.query('city=="sj"')[feature].copy()
                df['new_feature_sj'] = df.query('city=="sj"')[feature]

                # Plot two lineplots if any nans
                sns.lineplot(x=df['date'], y=old_feature_sj, ax=ax[i, 0], color=ascent_hex_color, label='original')
                sns.lineplot(x=df['date'], y=df['new_feature_sj'].fillna(np.inf), ax=ax[i, 0], color=hex_grey_color,
                              label='modified')

                ax[i, 0].set_title('San Juan', fontsize=14)
                ax[i, 0].set_xlabel('')
                ax[i, 0].set_xlim([datetime.date(1990, 4, 30),
                                datetime.date(2010, 6, 25)])

                # Iquitos
                old_feature_iq = df.query('city=="iq"')[feature].copy()
                df['new_feature_iq'] = df.query('city=="iq"')[feature]

                sns.lineplot(x=df['date'], y=old_feature_iq, ax=ax[i, 1], color=ascent_hex_color, label='original')
                sns.lineplot(x=df['date'], y=df['new_feature_iq'].fillna(np.inf), ax=ax[i, 1], color=hex_grey_color,
                              label='modified')
                ax[i, 1].set_title('Iquitos', fontsize=14)
                ax[i, 1].set_xlabel('')
                ax[i, 1].set_xlim([datetime.date(1990, 4, 30),
                                datetime.date(2010, 6, 25)])

            f.savefig(f'../fig/Explore_with_missing_{figure_name}.png')

        # features = ['Mean air temperature forecast', 'Diurnal temperature range forecast']
        # for term in ['centroid', 'forecast', 'station', 'NCEP']:
        #     plot_with_missing(self.train_data, figure_name=term, term=term)

        # print(self.train_data.query('city=="sj"')['total_cases'])

        # Plot raw features as is (only ffil)
        # for term in ['centroid', 'forecast', 'station', 'NCEP']:
        #     plot_raw_features(term = term)

        test_ne_centroid = self.train_data['Pixel northeast of city centroid'].copy()
        print(self.train_data[test_ne_centroid.isnull()]['date'].values)


        def plot_charts_with_nans(df):

            feature = 'Pixel northeast of city centroid'
            df['feature_ffil'] = df[feature].ffill()

            line_with_nans = alt.Chart(df).mark_line().encode(
                x=alt.X('date:T'),
                y=alt.Y(f'{feature}:Q'),
                color=alt.value(COLORHEX_GREY)
            ).transform_filter(
                "datum.city !== 'iq'"
            )

            line_cleaned = alt.Chart(df).mark_line().encode(
                x=alt.X('date:T'),
                y=alt.Y(f'feature_ffil:Q'),
                color=alt.value(COLORHEX_ASCENT)
            ).transform_filter(
                "datum.city !== 'iq'"
            )

            chart_ffil = (line_cleaned + line_with_nans).encode(
                x=alt.X().title(""),
                y=alt.Y().title("Fill NA by last valid")
            ).properties(
                width=800,
                title=f'San Juan : {feature}'
            )

            chart = chart_ffil

            chart.save('../fig/lines.png')

        plot_charts_with_nans(self.train_data)
