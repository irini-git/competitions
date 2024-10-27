import pandas as pd
import os
import altair as alt
import logging

# TODO : log data to file not print to screen

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILE_BARCHART_LABELS = '../fig/labels_bar_chart.png'
FILE_BARCHART_LABELS_TEST = '../fig/labels_bar_chart_test.png'

class FluShotData:
    """
    Class responsible for Flu Shot Data
    """
    def __init__(self):
        self.df_labels, self.df_features = self.load_data()
        self.explore_features()
        # self.explore_labels(self.df_labels)

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

        for df in [df_train]:
            print(f'Dataset has {df.shape[0]} entries.')
            print(f'Dataset has columns : {df.columns.tolist()}.')

        # Column names for vaccines
        columns = ['h1n1_vaccine', 'seasonal_vaccine']
        self.explore_labels(df_train[columns])

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
        self.plot_bar_chart_train_data(df_counts_percentages)


    def plot_bar_chart_train_data(self, df):
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