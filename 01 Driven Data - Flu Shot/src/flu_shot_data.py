import pandas as pd
import os

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILE_BARCHART_LABELS = '../fig/labels_bar_chart.png'
FILE_BARCHART_LABELS_TEST = '../fig/labels_bar_chart_test.png'

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
        :return: raw train and test data
        """
        df_labels = pd.read_csv(FILENAME_INPUT_DATA_LABELS)

        # TODO load features
        # TODO load test data

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

        # Placeholder for data
        columns_ = ['received', 'count', 'percentage', 'vaccine']
        df_counts_percentages = pd.DataFrame()

        for c in columns:
            # Calculate counts and percentages
            counts = df[c].value_counts() # value counts
            # percentages_round = round(100*(df[c].value_counts(normalize=True)),1).astype(str) + '%' # percentage as str with % sign
            percentages = 100*df[c].value_counts(normalize=True) # percentage value as is

            # Combines counts and percentages
            df_temp = pd.concat([counts, percentages], axis=1).reset_index()

            # Add a column for which vaccine
            df_temp['vaccine'] = c

            df_temp.columns = columns_

            df_counts_percentages = pd.concat([df_temp, df_counts_percentages])

        print(df_counts_percentages)

        self.plot_bar_chart_train_data(df_counts_percentages)


    def plot_bar_chart_train_data(self, df):
        """
        Plot bar chart for labels
        :return: figure in
        """

        import altair as alt
        from vega_datasets import data

        source = data.barley()

        print(source)

        bars_test = alt.Chart(source).mark_bar().encode(
            x=alt.X('sum(yield):Q').stack('zero'),
            y=alt.Y('variety:N'),
            color=alt.Color('site')
        )

        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X('count:Q').stack('zero'),
            y=alt.Y('vaccine:N'),
            color=alt.Color('received')
        ).properties(
            width=800,
            height=200
        )

        text = alt.Chart(source).mark_text(dx=-15, dy=3, color='white').encode(
            x=alt.X('sum(yield):Q').stack('zero'),
            y=alt.Y('variety:N'),
            detail='site:N',
            text=alt.Text('sum(yield):Q', format='.1f')
        )

        chart_test = bars_test + text
        chart = bars

        # Save the image in the img folder
        chart_test.save(FILE_BARCHART_LABELS_TEST)
        chart.save(FILE_BARCHART_LABELS)




    def get_current_location(self):
        """
        Support function to return the current directly
        :return: prints the current directory
        """
        print(os.getcwd())