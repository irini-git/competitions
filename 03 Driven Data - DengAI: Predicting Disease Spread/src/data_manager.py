import pandas as pd
import time
import datetime
import logging
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from statsmodels.tsa.seasonal import seasonal_decompose

from networkx.algorithms.traversal import dfs_tree

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
        self.train_data_cleaned = self.feature_engineering(self.train_data)


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

    def explore_data(self):

        # Explore data
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.train_data.head(2))
            print(self.train_data.info())

        # Parse date column to datetime format
        self.train_data['date'] = pd.to_datetime(self.train_data['week_start_date'], format='%Y-%m-%d')

        # Print to screen
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.train_data['week_start_date'].head(3))

        # Check if chronological order
        self.train_data = self.train_data.sort_values(by='date')
        print(f"Is monotonic increasing : {self.train_data['date'].is_monotonic_increasing}")


        # Suppot Visualization (how to deal with missing values)

        # ----------------
        # Create an array with the colors you want to use
        colors = ["#b3b7bd", "#1457ba"]
        # Set custom color palette
        sns.set_palette(sns.color_palette(colors))

        # ----------------
        def plot_raw_features(term):
            """
            Support function to plot raw features as is
            :param term: parameter family, for example "station"
            :return: figures as png
            """
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

        def plot_charts_with_nans(df):
            """
            Support function to plot features as is (with missing values)
            and when replaced using different methods
            :param df: feature to explore
            :return: Chart as png
            """

            def support_plot_fillna(df, method, city):

                if method=='ffill':
                    df[f'feature_{method}'] = df.query('city==@city')[feature].ffill()
                elif method=='by_mean':
                    df[f'feature_{method}'] = df.query('city==@city')[feature].fillna(df[feature].mean())
                elif method=='interpolate_linear':
                    df[f'feature_{method}'] = df.query('city==@city')[feature].interpolate(method='linear')
                elif method=='interpolate_cubic':
                    df[f'feature_{method}'] = df.query('city==@city')[feature].interpolate(method='cubic')
                elif method=='to_zero':
                    df[f'feature_{method}'] = df.query('city==@city')[feature].replace(np.nan, 0)

                # Plots
                line_with_nans = alt.Chart(df).mark_line().encode(
                    x=alt.X('date:T'),
                    y=alt.Y(f'{feature}:Q'),
                    color=alt.value(COLORHEX_GREY)
                ).transform_filter(
                    f"datum.city == '{city}'"
                )

                line_cleaned = alt.Chart(df).mark_line().encode(
                    x=alt.X('date:T'),
                    y=alt.Y(f'feature_{method}:Q'),
                    color=alt.value(COLORHEX_ASCENT)
                ).transform_filter(
                    f"datum.city == '{city}'"
                )

                chart = (line_cleaned + line_with_nans).encode(
                    x=alt.X().title(""),
                    y=alt.Y().title(f"Fill NA : {method}")
                ).properties(
                    width=800,
                    title = {
                        "text": [f"{city}"],
                        "subtitle": [f"{feature}"]
                    }
                )

                return chart

            # Feature to plot
            for feature, city in list(itertools.product(['Pixel northeast of city centroid',
                                'Pixel northwest of city centroid',
                                'Pixel southeast of city centroid',
                                'Pixel southwest of city centroid',
                                'Total precipitation station satellite',
                                'Mean air temperature forecast',
                                'Average air temperature NCEP',
                                'Mean dew point temperature NCEP',
                                'Maximum air temperature NCEP',
                                'Total precipitation kg_per_m2 NCEP',
                                'Total precipitation mm NCEP',
                                'Mean specific humidity NCEP',
                                'Diurnal temperature range forecast',
                                'Average temperature station',
                                'Diurnal temperature range station',
                                'Maximum temperature station',
                                'Minimum temperature station',
                                'Total precipitation station station',
                                'Mean relative humidity NCEP'
                                ],
                            ['iq', 'sj'])):

                # Create charts for different methods
                chart_left1 = support_plot_fillna(df, method='ffill', city=city)
                chart_right1 = support_plot_fillna(df, method='interpolate_cubic',city=city)

                chart_left2 = support_plot_fillna(df, method='interpolate_linear',city=city)
                chart_right2 = support_plot_fillna(df, method='to_zero',city=city)

                # Upper and bottom charts (to better visualization)
                chart1 = chart_left1 | chart_right1
                chart2 = chart_left2 | chart_right2

                # Combine all charts together
                chart = alt.vconcat(chart1, chart2).configure_title(
                    anchor='start'
                )

                # Save chart as png
                chart.save(f'../fig/{feature}_{city}.png')

        # Explore features as is ---------------------------------
        # features = ['Mean air temperature forecast', 'Diurnal temperature range forecast']
        # for term in ['centroid', 'forecast', 'station', 'NCEP']:
        #     plot_with_missing(self.train_data, figure_name=term, term=term)

        # Explore missing values city-wise -----------------------
        # and choose the method how to deal with nans
        # plot_charts_with_nans(self.train_data)

        # View df columns
        # previpitation
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
             print(self.train_data.info())

        # Handle missing values (ffill) --------------------------------
        not_columns = ['city', 'year', 'weekofyear', 'week_start_date','total_cases', 'date']
        features_ffill = list(set(self.train_data.columns) - set(not_columns))

        for f in features_ffill:
            self.train_data[f] = self.train_data.groupby('city')[f].ffill()

        # Time features - engineering ---------------
        self.train_data['year'] = pd.DatetimeIndex(self.train_data['date']).year
        self.train_data['month'] = pd.DatetimeIndex(self.train_data['date']).month
        self.train_data['day'] = pd.DatetimeIndex(self.train_data['date']).day
        self.train_data['day_of_year'] = pd.DatetimeIndex(self.train_data['date']).dayofyear
        self.train_data['quarter'] = pd.DatetimeIndex(self.train_data['date']).quarter
        self.train_data['season'] = self.train_data['month'] % 12 // 3 + 1

        # Encode cyclical features - months, days

        def encode_cyclical_features(df, col, max_val):
            df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
            return df

        for feature, max_val in zip(['month', 'day', 'quarter', 'season'], [12, 31, 4, 4]):
            self.train_data = encode_cyclical_features(self.train_data, col=feature, max_val=max_val)

        # View df columns
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # print(self.train_data[columns_time].head(2))
            # print(self.train_data.info())
            # for c in columns_time:
            #    print(self.train_data[c].value_counts())


        # ----------

        def plot_correlation(df):

            # Correlation
            # Uncomment if you want to plot correlation

            fig, ax = plt.subplots(figsize=(15,10))

            corrMatrix = df[features_ffill].corr().abs()
            sns_plot = sns.heatmap(corrMatrix,
                                   annot=True,
                                   cmap="Greys",
                                   linewidths=0.5, linecolor='white',
                                   ax=ax, cbar=False)
            fig.savefig('../fig/Correlation.png', bbox_inches='tight')

            fig = sns_plot.get_figure()

            print('Highly correlated features', '-' * 20)

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                for column in corrMatrix:
                    values_test = [ind for v, ind in zip(corrMatrix[column].values, corrMatrix[column].index) if
                                   v > 0.95]
                    if column != ''.join(values_test):
                        print(f'{column} : {values_test}')

            return corrMatrix

        # Uncomment to plot correlation Matrix
        plot_correlation(self.train_data)


    def feature_engineering(self, df):
        """
        Function responsible for feature engineering for train and test data
        :return:
        """
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.info())
            print(df.columns)

        numeric_features = ['city',	'year',	'weekofyear']

        # Parse date column to datetime format
        df['date'] = pd.to_datetime(df['week_start_date'], format='%Y-%m-%d')

        return df


