import pandas as pd
import os
import altair as alt
import logging
import functools

# TODO : log data to file not print to screen

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILE_BARCHART_LABELS = '../fig/labelas_bar_chart.png'
FILE_BARCHART_FEATURES_RATING = '../fig/features_bar_chart_rating.png'

class FluShotData:
    """
    Class responsible for Flu Shot Data
    """
    def __init__(self):
        self.df_labels, self.df_features = self.load_data()
        self.explore_features()

    def load_data(self):
        """
        Load raw data (features and labels) from csv files.
        :return: raw train and test data
        """
        df_labels = pd.read_csv(FILENAME_INPUT_DATA_LABELS)
        df_features = pd.read_csv(FILENAME_INPUT_DATA_FEATURES)

        return df_labels, df_features

    def explore_features(self):
        """
        Explore features
        :return:  output to screen, charts as files
        """

        df_train = pd.merge(self.df_labels, self.df_features, on="respondent_id")

        columns_explore = df_train.columns.tolist()
        columns_explore.remove("respondent_id")
        print(columns_explore)

        print(df_train.columns)

        print(f'Dataset has {df_train.shape[0]} entries.')
        print(f'Dataset has columns : {df_train.columns.tolist()}.')

        # Check if empty records
        # for c in df_train.columns:
        #     print(c, df_train[c].isna().sum())

        # plot distibution of values 


        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #       print(df_train[columns_explore].describe())

        # Feature engineering
        # 1. Features with rating
        # Create a new feature (str) with description

        # Scale for concern
        map_concern = {0.0: '0 - none',
                       1.0: '1 - a little',
                       2.0: '3 - somewhat',
                       3.0: '4 - a lot'}
        # Combine the map for concern with h1n1_concern
        df_train['h1n1_concern_desc'] = df_train['h1n1_concern'].map(map_concern)

        # Scale for knowledge
        map_knowledge = {0.0: '0 - none',
                       1.0: '1 - a little',
                       2.0: '4 - a lot'}
        # Combine the map for concern with h1n1_concern
        df_train['h1n1_knowledge_desc'] = df_train['h1n1_knowledge'].map(map_knowledge)

        # Scale for effective, risk and sick
        map_effective = {1.0: '0 - none',
                         2.0: '1 - a little',
                         3.0: '2 - dont know',
                         4.0: '3 - somewhat',
                         5.0: '4 - a lot'}
        # Apply the map
        for c in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                  'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc']:
            # Replace numeric values by description
            df_train[f'{c}_desc'] = df_train[c].map(map_effective)
        # Replace missing values NaN by 'dont know'
        for c in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                  'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc',
                  'h1n1_concern', 'h1n1_knowledge']:
            df_train[f'{c}_desc'] = df_train[f'{c}_desc'].fillna('2 - dont know')


        # -------------------------------------------------------
        # Explore labels as is, without features
        columns = ['h1n1_vaccine', 'seasonal_vaccine']
        # self.explore_labels(df_train[columns])

        # self.plot_bar_chart_h1n1_concern_kn(df_train)
        self.plot_stacked_bar(df_train)

    def plot_stacked_bar(self, df):
        from vega_datasets import data

        source = data.barley()

        df1 = df['h1n1_concern_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df1['feature'] = '4 concern'

        df2 = df['h1n1_knowledge_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df2['feature'] = '3 knowledge'

        df3 = df['opinion_h1n1_vacc_effective_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df3['feature'] = '0 vaccine effectiveness'

        df4 = df['opinion_h1n1_risk_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df4['feature'] = '2 getting sick without vaccine'

        df5 = df['opinion_h1n1_sick_from_vacc_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df5['feature'] = '1 getting sick from vaccine'

        df6 = df['opinion_seas_risk_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df6['feature'] = '2 getting sick without vaccine'

        df7 = df['opinion_seas_vacc_effective_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df7['feature'] = '0 vaccine effectiveness'

        df8 = df['opinion_seas_sick_from_vacc_desc'].value_counts().rename_axis('rating').reset_index(name='counts')
        df8['feature'] = '1 getting sick from vaccine'

        dfs_h1n1 = [df3, df4, df5, df1, df2]
        dfs_seas = [df6, df7, df8]
        source_h1n1 = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_h1n1)
        source_seas = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_seas)

        chart_h1n1 = alt.Chart(source_h1n1, title='Opinion on H1N1 flu vaccine').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y('feature:N').title(''),
            color = alt.Color('rating',
                              legend=alt.Legend(title="Ratings"),
                              scale=alt.Scale(scheme='redyellowgreen')
                              )
            )

        chart_seas = alt.Chart(source_seas, title='Opinion on seasonal vaccine').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y('feature:N').title(''),
            color = alt.Color('rating',
                              legend=alt.Legend(title="Ratings"),
                              scale=alt.Scale(scheme='redyellowgreen')
                              )
        )

        chart = chart_h1n1 | chart_seas

        chart.save(FILE_BARCHART_FEATURES_RATING)


    def explore_labels(self, df):
        """
        How many people receive H1N1 / seasonal flu vaccine and how many do not?
        To find out, let's explore the distribution of values in train dataset as is, without other factors.
        The function explores train dataset, calculates stats and visualize
        :param df: dataframe to explore
        :return: output to screen, charts as files
        """

        # Placeholder for data
        columns_ = ['vaccinated', 'count', 'percentage', 'vaccine']
        df_counts_percentages = pd.DataFrame()

        for c in df.columns:
            # Calculate counts and percentages
            counts = df[c].value_counts() # value counts

            # Values for a log file, easier to read
            # percentages_round = round(100*(df[c].value_counts(normalize=True)),1).astype(str) + '%' # percentage as str with % sign

            percentages = 100*df[c].value_counts(normalize=True) # percentage value as is

            # Combines counts and percentages
            df_temp = pd.concat([counts, percentages], axis=1).reset_index()

            # Add a column for which vaccine
            df_temp['vaccine'] = c

            # Rename columns
            df_temp.columns = columns_

            # Replace values for vaccinated to Yes/No for appealing visual, current values: 1/0.
            df_temp['vaccinated'] = df_temp['vaccinated'].map({0: 'No', 1: 'Yes'})

            # Concat dataframes to one
            df_counts_percentages = pd.concat([df_temp, df_counts_percentages])

        # TODO : output to log file
        print(df_counts_percentages)

        # Plot bar chart
        self.plot_bar_chart_labels(df_counts_percentages)


    def plot_bar_chart_labels(self, df):
        """
        Plot bar chart for labels
        :return: figure in
        """

        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X('count:Q').title('counts').stack('zero'),
            y=alt.Y('vaccine:N').title(''),
            color='vaccinated'
        ).properties(
            width=800,
            height=200
        )

        text = bars.mark_text(
            align='left',
            baseline='middle',
            color='white',
            dx=10, dy=0  # Nudges text to the right so it doesn't appear on top of the bar
        ).encode(
            text='count:Q'

        )

        chart = bars + text

        # Save the image in the fig folder
        chart.save(FILE_BARCHART_LABELS)


    def get_current_location(self):
        """
        Support function to return the current directly
        :return: prints the current directory
        """
        print(os.getcwd())