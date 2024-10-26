import pandas as pd
import os

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

class FluShotData:
    """
    Class responsible for Flu Shot Data
    """
    def __init__(self):
        self.df_train = self.load_data()
        self.explore_train_data(self.df_train)

    def load_data(self):
        """
        Load raw data (features and labels) from csv files.
        :return:
        """
        df_labels = pd.read_csv(FILENAME_INPUT_DATA_LABELS)
        return df_labels

    def explore_train_data(self, df):
        """
        Explore train dataset, calculate stats and visualize
        :param df: dataframe to explore
        :return: output to screen, charts as files
        """
        print('Exploring dataset', '-'*10)
        print(f'Dataset has columns : {df.columns.tolist()}.')
        print(f'Dataset has {df.shape[0]} entries.')

        columns = ['h1n1_vaccine', 'seasonal_vaccine']

        # How many people receive H1N1 / seasonal flu vaccine?
        # To do this, let's explore the distribution of values in train dataset

        for c in columns:
            # Calculate counts and percentages
            counts = df[c].value_counts()
            percentages = round(100*(df[c].value_counts(normalize=True)),1).astype(str) + '%'

            # Combines counts and percentages
            df_counts_percentages = pd.concat([counts, percentages], axis=1, keys=['count', 'percentage'])

            print(df_counts_percentages)


    def get_current_location(self):
        """
        Support function to return the current directly
        :return: prints the current directory
        """
        print(os.getcwd())