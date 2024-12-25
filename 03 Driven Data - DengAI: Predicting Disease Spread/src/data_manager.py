import pandas as pd
import time
import datetime
import logging
import altair as alt

# Constants
TEST_DATA_FEATURES = '../data/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv'
TRAINING_DATA_FEATURES = '../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv'
TRAINING_DATA_LABELS = '../data/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv'

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

        # Simplify column names

        train_data.rename(columns={"station_diur_temp_rng_c": "Diurnal temperature range station",
                                   "station_precip_mm": "Total precipitation station station",
                                   "station_min_temp_c": "Minimum temperature station",
                                   "station_max_temp_c": "Maximum temperature station",
                                   "precipitation_amt_mm": "Total precipitation satellite",
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
                                   "ndvi_nw":"Pixel northwest of city centroid"
                                   },
                          inplace=True)

        print(train_data.columns)


        return train_data, test_data_features


    def explore_data(self):
        """
        Data exploration, search for patterns and insights.
        :return: ideas for feature engineering
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.train_data.head(2))
            print(self.train_data.info())
            for c in self.train_data.columns:
                print(self.train_data[c].value_counts().head(2))

        def plot_scatter_cites():

            chart = alt.Chart(self.train_data).mark_line(point=True).encode(
                x='year:N',
                y='total_cases:Q',
                color='city:N'
            )

            chart.save('../fig/explore_002.png')

        plot_scatter_cites()