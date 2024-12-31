
import pandas as pd
import time
import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
import logging
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose

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


        return train_data, test_data_features

    def explore_data(self):

        # Explore data
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # print(self.train_data.head(2))
            # print(self.train_data.info())

        # Parse date column to datetime format
        self.train_data['date'] = pd.to_datetime(self.train_data['week_start_date'], format='%Y-%m-%d')

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

                # Combine charts
                chart = alt.vconcat(chart1, chart2).configure_title(
                    anchor='start'
                )

                # Save chart as png
                chart.save(f'../fig/{feature}_{city}.png')

        # ----------
        def plot_correlation(df, components=False):

            # Correlation
            # Uncomment if you want to plot correlation

            if components:
                columns_to_plot = [c for c in self.train_data if any(['seasonal' in c, 'trend' in c])]
            else:
                not_columns = ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases', 'date']
                columns_to_plot = list(set(df.columns) - set(not_columns))

            fig, ax = plt.subplots(figsize=(15,10))

            corrMatrix = df[columns_to_plot].corr().abs()
            sns.heatmap(corrMatrix,
                        annot=True,
                        cmap="Greys",
                        linewidths=0.5, linecolor='white',
                        ax=ax, cbar=False)
            fig.savefig(f'../fig/Correlation_components_{components}.png', bbox_inches='tight')
            plt.close()

            print('Highly correlated features', '-' * 20)

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                for column in corrMatrix:
                    values_test = [ind for v, ind in zip(corrMatrix[column].values, corrMatrix[column].index) if
                                   v > 0.95]
                    if column != ''.join(values_test):
                        print(f'{column} : {values_test}')

            return corrMatrix

        # Actual exploration, uncomment to generate charts

        # Explore features as is ---------------------------------
        # for term in ['centroid', 'forecast', 'station', 'NCEP']:
        #       plot_raw_features(term)

        # Explore missing values city-wise -----------------------
        # and choose the method how to deal with nans
        # plot_charts_with_nans(self.train_data)

        # Uncomment to plot correlation Matrix
        # plot_correlation(self.train_data, components=False)

        # ---------------------
        # Explore features vs label
        # Handle missing values (ffill) --------------------------------
        not_columns = ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases', 'date']
        features_ffill = list(set(self.train_data.columns) - set(not_columns))

        for f in features_ffill:
            self.train_data[f] = self.train_data.groupby('city')[f].ffill()

        # Time features - engineering ------------------------------
        self.train_data['year'] = pd.DatetimeIndex(self.train_data['date']).year
        self.train_data['month'] = pd.DatetimeIndex(self.train_data['date']).month
        self.train_data['day'] = pd.DatetimeIndex(self.train_data['date']).day
        self.train_data['day_of_year'] = pd.DatetimeIndex(self.train_data['date']).dayofyear
        self.train_data['quarter'] = pd.DatetimeIndex(self.train_data['date']).quarter
        self.train_data['season'] = self.train_data['month'] % 12 // 3 + 1

        # Encode cyclical features - months, days
        def encode_cyclical_features(df_, col, max_val_):
            df_[col + '_sin'] = np.sin(2 * np.pi * df_[col] / max_val_)
            df_[col + '_cos'] = np.cos(2 * np.pi * df_[col] / max_val_)
            return df_

        for feature, max_val in zip(['month', 'day', 'quarter', 'season'], [12, 31, 4, 4]):
            self.train_data = encode_cyclical_features(self.train_data, col=feature, max_val_=max_val)

        # Preview --------------------------------
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.train_data.info())
        #    print(self.train_data['date'].head(6))

        def plot_city_vs_label(df):

            # COLORHEX_GREY = '#767676'
            # COLORHEX_ASCENT = '#ff4d00'
            range_ = ['#767676', '#ff4d00']
            domain = ['sj', 'iq']

            chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X('date', title=''),
                y=alt.Y('total_cases:Q', title='Total cases').stack(None),
                color=alt.Color("city",
                                scale=alt.Scale(domain=domain, range=range_),
                                legend=None),
                ).properties(
                    width=800,
                    height=300,
                    title={
                        "text": [f"Dengue cases {self.train_data.year.min()} - {self.train_data.year.max()}"],
                        "subtitle": ["in San Juan (grey) and Iquitos (orange)"]
                    }
                ).configure_title(
                    anchor='start'
                )

            chart.save('../fig/city_vs_label.png')

        def plot_calendar_heatmap(location):
        # Heatmap per city

            # Up value for the scale
            df = self.train_data.query('city==@location').copy()
            up_value = round(df.groupby(by=["year", "month"])['total_cases'].sum().max(),-1)

            # City name is plain English for the title
            if location == 'iq':
                city_ = 'Iquitos, Peru'
            else:
                city_ = 'San Juan, Puerto Rico'

            chart = alt.Chart(df, title=f'Calendar view for dengue cases in {city_}').mark_rect().encode(
                x=alt.X('year:O', title=''),
                y=alt.Y('month(date):O', title=''),
                color=alt.Color(
                    'total_cases:Q',
                    scale=alt.Scale(scheme='reds',
                    domain=(0, up_value/12))),
                tooltip=[
                    alt.Tooltip('year:O', title='Year'),
                    alt.Tooltip('month:O', title='Month'),
                    alt.Tooltip('count(total_cases):Q', title='total_cases')
                ]).properties(width=600, height=300)


            chart.save(f'../fig/cities_heatmap_{location}.png')

        def plot_feature_vs_label(feature, location):

            # Up value for the scale
            df = self.train_data.query('city==@location').copy()

            # City name is plain English for the title
            if location == 'iq':
                city_ = 'Iquitos, Peru'
            else:
                city_ = 'San Juan, Puerto Rico'

            base2 = alt.Chart(df).encode(
                alt.X('date:T', axis=alt.Axis(title=None))
            )

            line3 = base2.mark_line(stroke=COLORHEX_GREY, interpolate='monotone').encode(
                alt.Y('total_cases',
                      axis=alt.Axis(
                          title='Dengue cases',
                          titleColor=COLORHEX_GREY))
            )

            line2 = base2.mark_line(stroke='#5276A7', interpolate='monotone').encode(
                alt.Y(feature,
                      axis=alt.Axis(title=f'{feature}', titleColor='#5276A7'))
            )

            chart = alt.layer(line2, line3).resolve_scale(
                y='independent'
            ).properties(width=700, height=300,
                         title={"text": [f"Dengue cases and {feature} in {city_}"]}
            ).configure_title(
                anchor='start'
            )

            chart.save(f'../fig/{feature}_{location}.png')


        def resample(term, location):

            # City name is plain English for the title
            if location == 'iq':
                city_ = 'Iquitos, Peru'
            else:
                city_ = 'San Juan, Puerto Rico'

            # Features to plot
            columns_to_visualize = [c for c in self.train_data if term in c]

            # Filter for location
            df = self.train_data.query('city==@location').copy()

            fig, ax = plt.subplots(ncols=2, nrows=len(columns_to_visualize), sharex=True, figsize=(14, 8))
            fig.suptitle(f'{city_} : original (left) and monthly (right)')

            # ---------------
            for idx, feature in enumerate(columns_to_visualize):
                text_annotation = feature.split(' ')
                new_text_annotation = ' '.join(text_annotation[:2]) + '\n' + ' '.join(text_annotation[2:])

                ax[idx, 0].bar(df['date'], df[feature], width=5, color=COLORHEX_GREY)
                ax[idx, 0].set_ylabel(f'{new_text_annotation}', fontsize=8, rotation='horizontal', ha='right')

                resampled_df = df[['date', feature]].resample('ME', on='date').sum().reset_index(drop=False)
                ax[idx, 1].bar(resampled_df['date'], resampled_df[feature], width=10, color=COLORHEX_GREY)

            # ---------------

            fig.savefig(f'../fig/resampled_{term}_{location}.png')

        def create_decompositions(feature):
            """
            Create support columns responsible for feature decomposition
            :param feature: column in df
            :return: updated train data with trends and seasonal components
            # Replace train by df
            """

            # Filter for location
            df_iq = self.train_data.query('city=="iq"').copy()
            df_sj = self.train_data.query('city=="sj"').copy()

            # Decomposition
            res_iq = seasonal_decompose(df_iq[feature], model='additive', extrapolate_trend='freq', period=52)
            res_sj = seasonal_decompose(df_sj[feature], model='additive', extrapolate_trend='freq', period=52)

            # Combine in one column based on index
            trend = res_sj.trend.combine_first(res_iq.trend)
            seasonal = res_sj.seasonal.combine_first(res_iq.seasonal)

            # Add trend and seasonal columns
            self.train_data = pd.concat([self.train_data, trend, seasonal], axis=1)

            # Rename columns
            self.train_data.rename(columns={'trend': f'{feature} trend',
                                            'seasonal' : f'{feature} seasonal'}
                                            , inplace=True)


        # Uncomment for decomposition
        # for feature in features_ffill:
        #     create_decompositions(feature=feature)

        # plot_correlation(self.train_data, components=True)

        def perform_seasonal_decomposition(feature, location):

            # City name is plain English for the title
            if location == 'iq':
                city_ = 'Iquitos, Peru'
            else:
                city_ = 'San Juan, Puerto Rico'

            # Filter for location
            df = self.train_data.query('city==@location').copy()
            df = df.set_index('date')

            # Decomposition
            res = seasonal_decompose(df[feature], model='additive',
                                     extrapolate_trend='freq', period=52)

            x_min = df.index.values[0]
            x_max = df.index.values[-1]

            fig, ax = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(16, 8))
            plt.xlim(x_min, x_max)

            ax[0].set_title(f'Decomposition of {feature} in {city_}', fontsize=14)

            res.observed.plot(ax=ax[0], legend=False, color=COLORHEX_GREY)
            ax[0].set_ylabel('Observed', fontsize=12)

            res.trend.plot(ax=ax[1], legend=False, color=COLORHEX_GREY)
            ax[1].set_ylabel('Trend', fontsize=12)

            res.seasonal.plot(ax=ax[2], legend=False, color=COLORHEX_GREY)
            ax[2].set_ylabel('Seasonal', fontsize=12)

            res.resid.plot(ax=ax[3], legend=False, color=COLORHEX_GREY)
            ax[3].set_ylabel('Residual', fontsize=12)
            ax[3].tick_params(axis='x', rotation=0)
            ax[3].set_xlabel('')

            # Save chart to file
            fig.savefig(f'../fig/decomposition_{feature}_{location}.png')

            # Close figure
            plt.close()


        for term, city in list(itertools.product(['centroid', 'forecast', 'station', 'NCEP'],
                    ['iq', 'sj'])):
            # Resample to monthly
            # resample(term=term, location=city)
            pass

        # plot_city_vs_label(self.train_data)
        # plot_calendar_heatmap(location='iq')
        # plot_calendar_heatmap(location='sj')

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

            # plot_feature_vs_label(feature=feature, location=city)
            # resample(feature=feature, location=city)
            # perform_seasonal_decomposition(feature=feature, location=city)
            pass

        def plot_boxplot_time_vs_cases(df, time_period='year'):
            fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(16, 8))

            # Boxplots
            sns.boxplot(ax=ax[0], data=df.query('city=="sj"'), x=time_period, y='total_cases')
            sns.boxplot(ax=ax[1], data=df.query('city=="iq"'), x=time_period, y='total_cases')

            # Subtitles as y labels
            ax[0].set_ylabel('San Juan (Puerto Rico)')
            ax[1].set_ylabel('Iquitos (Peru)')

            # ax[0].set_xticks(df.year)
            ax[1].set_xlabel('')

            # despine top and right borders
            sns.despine(left=False, right=True, bottom=False, top=True)

            fig.suptitle(f'Dengue cases in Iquitos (Peru) and San Juan (Puerto Rico) in {df.year.min()}-{df.year.max()}')
            fig.savefig(f'../fig/box_plot_{time_period}.png')

        plot_boxplot_time_vs_cases(self.train_data, time_period='year')

    def feature_engineering(self, df):
        """
        Function responsible for feature engineering for train and test data
        - parse date column to datetime format
        - ensure chronological order
        - handle missing values (ffill)
        - time features : add new, encode cyclical
        :return:
        """

        # Parse date column to datetime format ---------------
        df['date'] = pd.to_datetime(df['week_start_date'], format='%Y-%m-%d')

        # Ensure chronological order ---------------
        df = df.sort_values(by='date')
        print(f"Is monotonic increasing : {df['date'].is_monotonic_increasing}")

        # Handle missing values (ffill) --------------------------------
        not_columns = ['city', 'year', 'weekofyear', 'week_start_date','total_cases', 'date']
        features_ffill = list(set(df.columns) - set(not_columns))

        for f in features_ffill:
            df[f] = df.groupby('city')[f].ffill()

        # Time features - engineering ------------------------------
        def create_time_features(df):
            """
            Create time series features
            """
            df = df.copy()
            df['year'] = pd.DatetimeIndex(df['date']).year
            df['month'] = pd.DatetimeIndex(df['date']).month
            df['day'] = pd.DatetimeIndex(df['date']).day
            df['day_of_year'] = pd.DatetimeIndex(df['date']).dayofyear
            df['quarter'] = pd.DatetimeIndex(df['date']).quarter
            df['season'] = df['month'] % 12 // 3 + 1
            return df

        df = create_time_features(df)

        # Encode cyclical features - months, days
        def encode_cyclical_features(df_, col, max_val_):
            df_[col + '_sin'] = np.sin(2 * np.pi * df_[col] / max_val_)
            df_[col + '_cos'] = np.cos(2 * np.pi * df_[col] / max_val_)
            return df_

        for feature, max_val in zip(['month', 'day', 'quarter', 'season'], [12, 31, 4, 4]):
            df = encode_cyclical_features(df, col=feature, max_val_=max_val)

        def create_decompositions(df):
            """
            Create support columns responsible for feature decomposition for selected items
            Based on prev analysis we only need below, other are highly correlated

                - Diurnal temperature range station trend
                - Pixel northwest of city centroid trend
                - Mean specific humidity NCEP trend
                - Total precipitation mm NCEP trend

                - Average air temperature NCEP seasonal
                - Minimum air temperature NCEP seasonal
                - Mean dew point temperature NCEP seasonal
                - Total precipitation mm NCEP seasonal

            :param feature: column in df
            :return: df data with trends and seasonal components
            """

            # Filter for location
            df_iq = df.query('city=="iq"').copy()
            df_sj = df.query('city=="sj"').copy()

            # Trend --------------
            for feature in ['Diurnal temperature range station',
                            'Pixel northwest of city centroid',
                            'Mean specific humidity NCEP',
                            'Total precipitation mm NCEP']:
                # Decomposition
                res_iq = seasonal_decompose(df_iq[feature], model='additive', extrapolate_trend='freq', period=52)
                res_sj = seasonal_decompose(df_sj[feature], model='additive', extrapolate_trend='freq', period=52)

                # Combine in one column based on index
                trend = res_sj.trend.combine_first(res_iq.trend)

                # Add trend and seasonal columns
                df = pd.concat([df, trend], axis=1)

                # Rename columns
                df.rename(columns={'trend': f'{feature} trend'}, inplace=True)

            # Seasonal --------------
            for feature in ['Average air temperature NCEP',
                            'Minimum air temperature NCEP',
                            'Mean dew point temperature NCEP',
                            'Total precipitation mm NCEP']:
                # Decomposition
                res_iq = seasonal_decompose(df_iq[feature], model='additive', extrapolate_trend='freq', period=52)
                res_sj = seasonal_decompose(df_sj[feature], model='additive', extrapolate_trend='freq', period=52)

                # Combine in one column based on index
                seasonal = res_sj.seasonal.combine_first(res_iq.seasonal)

                # Add trend and seasonal columns
                df = pd.concat([df, seasonal], axis=1)

                # Rename columns
                df.rename(columns={'seasonal': f'{feature} seasonal'}, inplace=True)

            return df

        # Create seasonal and trend decomposition for selected features
        df = create_decompositions(df)

        # Transform date to index
        df = df.set_index('date')

        numeric_raw_features = ['year', 'weekofyear',
                            'Pixel northeast of city centroid', 'Pixel northwest of city centroid',
                            'Pixel southeast of city centroid', 'Pixel southwest of city centroid',
                            'Mean air temperature forecast', 'Average air temperature NCEP',
                            'Maximum air temperature NCEP', 'Minimum air temperature NCEP',
                            'Total precipitation kg_per_m2 NCEP', 'Mean relative humidity NCEP',
                            'Total precipitation mm NCEP', 'Mean specific humidity NCEP',
                            'Diurnal temperature range forecast', 'Average temperature station',
                            'Diurnal temperature range station', 'Maximum temperature station',
                            'Minimum temperature station']

        columns_time = ['day_of_year', 'season', 'month_sin', 'month_cos',
                        'day_sin', 'day_cos', 'season_sin', 'season_cos', 'month', 'quarter']

        # Features for components
        columns_components = ['Total precipitation mm NCEP trend',
                            'Total precipitation mm NCEP seasonal',
                            'Mean dew point temperature NCEP seasonal',
                            'Minimum air temperature NCEP seasonal',
                            'Average air temperature NCEP seasonal',
                            'Mean specific humidity NCEP trend',
                            'Pixel northwest of city centroid trend',
                            'Diurnal temperature range station trend']

        FEATURES = ['city'] + numeric_raw_features + columns_components + columns_time
        TARGET = ['total_cases']

        df = df[FEATURES + TARGET]

        # # compute the correlations
        # sj_correlations = sj_train_features.corr()
        # iq_correlations = iq_train_features.corr()
        #
        # def plot_correlations(df, city):
        #     fig, ax = plt.subplots(figsize=(15, 15))
        #     df.total_cases.drop('total_cases').sort_values(ascending=False).plot.barh(ax=ax)
        #     plt.tight_layout()
        #     fig.savefig(f'../fig/corr_data_drive_{city}.png')
        #
        # plot_correlations(iq_correlations, 'iq')
        # plot_correlations(sj_correlations, 'sj')

        return df

    def create_model_HistGradientBoostingRegressor(self, df):
        """
        Creates model
        :return:
        """

        # Split cities
        # Test to have one year of data
        # locate the latest date and date minus 12 m

        df_iq = df.query('city=="iq"').drop('city', axis='columns')
        df_sj = df.query('city=="sj"').drop('city', axis='columns')

        # Preview --------------------------------
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # print(df.info())
            # print(df.columns)

        # Define numeric features
        # use different set of numeric features for cities
        numeric_features = list(set(df_iq.columns.values) - set(['total_cases']))

        # Define pipelines for numeric and categorical features -------------
        numeric_transformer = Pipeline(
            steps=[
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

        # Make our ColumnTransformer
        # Set remainder="passthrough" to keep the columns in our feature table which do not need any preprocessing.
        col_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
            ],
            remainder='passthrough'
        )


        def split_test_train(df):

            df = df[numeric_features + ['total_cases']]

            X = df.iloc[:, 0:-1]
            y = df.iloc[:, -1]

            tss = TimeSeriesSplit(n_splits=5)

            for train_index, test_index in tss.split(X):
                 X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                 y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            return X_train, y_train, X_test, y_test

        X_train_sj, y_train_sj, X_test_sj, y_test_sj = split_test_train(df_sj)
        X_train_iq, y_train_iq, X_test_iq, y_test_iq = split_test_train(df_iq)

        # ----------------
        def plot_train_test(train, test, city):
            train = train.to_frame(name='total_cases')
            test = test.to_frame(name='total_cases')

            fig, ax = plt.subplots(figsize=(15,5))
            train['total_cases'].plot(ax=ax, label='Training Set', color=COLORHEX_GREY)
            test['total_cases'].plot(ax=ax, label='Train Set', color=COLORHEX_ASCENT)
            ax.axvline(test.index[0], color='black', ls='--')

            plt.ylim(bottom=0)
            plt.xlim(left=train.index[0], right=test.index[-1])
            plt.xlabel('')
            plt.xticks(rotation=0)

            ax.spines[['top', 'right']].set_visible(False)

            plt.title(f'Data Train/Test Split for {city}', loc='left')
            plt.close()

            fig.savefig(f'../fig/train_test_{city}.png')

        # plot_train_test(y_train_sj, y_test_sj, 'sj')
        # plot_train_test(y_train_iq, y_test_iq, 'iq')

        # ------------

        # Run model per location

        def model_city(X_train, y_train, X_test, y_test):
            # model = xgb.XGBRegressor(n_estimators=1000,
            #                        early_stopping_rounds=50)

            param_grid = {
                'model__learning_rate': [0.01, 0.1],
                'model__max_iter': [100],
                'model__max_leaf_nodes': [10],
                'model__l2_regularization': [1.0]
            }

            classifier = HistGradientBoostingRegressor()

            # Make a pipeline
            main_pipe = Pipeline(
                steps=[
                    ("preprocessor", col_transformer),  # <-- this is the ColumnTransformer we created
                    ("model", classifier)])

            grid_search = GridSearchCV(main_pipe, param_grid, cv=2, verbose=4)

            grid_search.fit(X_train, y_train)

            result = permutation_importance(grid_search, X_train, y_train, n_repeats=10,
                                            random_state=0)

            # Create pandas DataFrame for feature importance
            data_fi = {'mean' : result.importances_mean,
                    'std' : result.importances_std}
            df_fi = pd.DataFrame(data_fi, index=X_train.columns.values)
            cols_exclude = df_fi.index[df_fi.eq(0).all(axis=1)].to_list()
            print(cols_exclude)
            print('-'*10)

            for i in result.importances_mean.argsort()[::-1]:
                if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
                    print(f"{X_train.columns.values[i]:<8}"
                          f" {result.importances_mean[i]:.3f}"
                          f" +/- {result.importances_std[i]:.3f}")

            print('Best Grid Search Parameters :', grid_search.best_params_)
            print('Best Grid Search Score : ', grid_search.best_score_)

            # Predict
            y_pred = grid_search.predict(X_test).astype(int)

            score = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f'RMSE Score on Test set: {score:0.2f}')

            return y_pred

            # -----------

        #
        print('SJ ----------- ')
        y_preds_sj = model_city(X_train_sj, y_train_sj, X_test_sj, y_test_sj)
        print('IQ ----------- ')
        y_preds_iq = model_city(X_train_iq, y_train_iq, X_test_iq, y_test_iq)

        # Save predictions
        X_test_sj['y_pred'] = y_preds_sj
        X_test_sj['y_test'] = y_test_sj
        X_test_sj.to_csv('../data/X_test_sj.csv')

        X_test_iq['y_pred'] = y_preds_iq
        X_test_iq['y_test'] = y_test_iq
        X_test_iq.to_csv('../data/X_test_iq.csv')

        # def plot_feature_importance():
        #     fig, ax = plt.subplots(figsize=(15,15))
        #     fi.sort_values('importance').plot(ax=ax,
        #                                       kind='barh',
        #                                       title='Feature Importance')
        #     plt.tight_layout()
        #     plt.close()
        #     fig.savefig('../fig/feature_importance.png')
        #
        # plot_feature_importance()

        return

    def load_predictions(self):

        X_test_sj = pd.read_csv('../data/X_test_sj.csv', index_col='date')
        X_test_sj.index = pd.to_datetime(X_test_sj.index)

        X_test_iq = pd.read_csv('../data/X_test_iq.csv', index_col='date')
        X_test_iq.index = pd.to_datetime(X_test_iq.index)

        def plot_predictions_vs_test(df, city):# plot sj

            fig, ax = plt.subplots(figsize=(15, 5))
            df['y_pred'].plot(ax=ax, label='Predictions', color=COLORHEX_ASCENT)
            df['y_test'].plot(ax=ax, label='Test', color=COLORHEX_GREY)

            plt.ylim(bottom=0)
            plt.xlim(left=df.index[0], right=df.index[-1])
            plt.xlabel('')
            plt.legend(loc='upper left')
            plt.xticks(rotation=0)

            ax.spines[['top', 'right']].set_visible(False)

            plt.title(f'Predictions vs test for {city}', loc='left')
            plt.close()

            fig.savefig(f'../fig/pred_test_{city}.png')

        # plot_predictions_vs_test(X_test_sj, 'sj')
        # plot_predictions_vs_test(X_test_iq, 'iq')


