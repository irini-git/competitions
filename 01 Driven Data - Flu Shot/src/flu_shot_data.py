from pyexpat import features

import numpy as np
import pandas as pd
import os
import altair as alt
import logging
import functools

# TODO : log data to file not print to screen

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILE_BARCHART_LABELS = '../fig/labels_bar_chart.png'
FILE_BARCHART_FEATURES_RATING = '../fig/features_bar_chart_rating.png'
FILE_BARCHART_FEATURES_BEHAVIOURAL = '../fig/features_bar_chart_behavioral.png'
FILE_BARCHART_FEATURES_DOCTOR_REC = '../fig/features_bar_chart_doctor_recommendation.png'
FILE_BARCHART_FEATURES_HEALTH = '../fig/features_bar_chart_health.png'
FILE_BARCHART_FEATURES_PERSONAL = '../fig/features_bar_chart_personal.png'

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
        # 1. Features with rating -----------------------------------
        # Create a new feature (str) with description and a new int feature
        # Features with description to be used for visual exploration,
        # Features with int values to be used for modeling.

        # Different features might have different ratings / scales
        # Scale for concern
        map_concern_desc = {0.0: '0 - none',
                       1.0: '1 - a little',
                       2.0: '3 - somewhat',
                       3.0: '4 - a lot'}
        map_concern_int = {0.0: 0, 1.0: 1, 2.0: 3, 3.0: 4}
        # Combine the map for concern with h1n1_concern
        df_train['h1n1_concern_desc'] = df_train['h1n1_concern'].map(map_concern_desc)
        df_train['h1n1_concern_int'] = df_train['h1n1_concern'].map(map_concern_int)

        # Scale for knowledge
        map_knowledge_desc = {0.0: '0 - none',
                       1.0: '1 - a little',
                       2.0: '4 - a lot'}
        map_knowledge_int = {0.0: 0, 1.0: 1, 2.0: 4}
        # Combine the map for concern with h1n1_concern
        df_train['h1n1_knowledge_desc'] = df_train['h1n1_knowledge'].map(map_knowledge_desc)
        df_train['h1n1_knowledge_int'] = df_train['h1n1_knowledge'].map(map_knowledge_int)

        # Scale for effective, risk and sick
        map_effective_desc = {1.0: '0 - none',
                         2.0: '1 - a little',
                         3.0: '2 - dont know',
                         4.0: '3 - somewhat',
                         5.0: '4 - a lot'}
        map_effective_int = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}

        # Apply the map
        for c in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                  'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc']:
            # Replace numeric values by description
            df_train[f'{c}_desc'] = df_train[c].map(map_effective_desc)
            df_train[f'{c}_int'] = df_train[c].map(map_effective_int)

        # Replace missing values NaN by 'dont know'
        for c in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                  'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc',
                  'h1n1_concern', 'h1n1_knowledge']:
            df_train[f'{c}_desc'] = df_train[f'{c}_desc'].fillna('2 - dont know')
            df_train[f'{c}_int'] = df_train[f'{c}_int'].fillna(2)
            df_train[f'{c}_int'] = df_train[f'{c}_int'].astype(int)


        # 2. Behavioral features Y/N
        # Create a new feature (str) with Y for 1, N for 0 and 'No response' for NaN
        # Create a new feature (int) with 2 for 1.0, 0 for 0.0 (dont know), and 1 for NaN
        # We address missing values in responses for behavioural features adding 'dont know' option.
        # Other technics are also possible, like imputation, deletion, subgroup analysis.

        features_behavioral = ['behavioral_antiviral_meds', 'behavioral_avoidance',
                               'behavioral_face_mask', 'behavioral_wash_hands',
                               'behavioral_large_gatherings', 'behavioral_outside_home',
                               'behavioral_touch_face']

        def float_to_word(w):
            if w == 0.0:
                return 'No'
            elif w == 1.0:
                return 'Yes'
            else:
                return 'No response'

        def float_to_int(w):
            if w == 0.0:
                return 0
            elif w == 1.0:
                return 2
            else:
                return 1

        # Apply the mapping function for behavioral features and return stats
        for f in features_behavioral:
            df_train[f'{f}_desc'] = df_train[f].apply(float_to_word)
            df_train[f'{f}_int'] = df_train[f].apply(float_to_int)
            # print output to log file
            # print(100*df_train[f'{f}_int'].value_counts()/df_train.shape[0])
            # print('-'*10)


        # 3. Chronic condition and doctors recommendation
        # Create a new feature (str) with Y for 1, N for 0 and 'No response' for NaN

        features_medical = ['doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition']

        for f in features_medical:
            df_train[f'{f}_desc'] = df_train[f].apply(float_to_word)

        features_health = ['chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance']

        for f in features_health:
            df_train[f'{f}_desc'] = df_train[f].apply(float_to_word)

        features_personal =['employment_status', 'rent_or_own', 'marital_status', 'sex']

        for f in features_personal:
            print(df_train[f].isnull().sum())
            print(100 * df_train[f].value_counts() / df_train.shape[0])
            print('-' * 10)

        # -------------------------------------------------------
        # Explore labels as is, without features
        columns = ['h1n1_vaccine', 'seasonal_vaccine']
        self.explore_labels(df_train[columns])

        self.plot_stacked_bar_ratings_behaviour_medical(df_train)

    def plot_stacked_bar_ratings_behaviour_medical(self, df):
        """
        Visualisation of rating-like features in train data.
        Leading numbers added to descriptions for visual purpose to align the graphs.
        :param df: train dataframe with all features
        :return: bar chart saved as png file
        """

        # Ratings ------------------------
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

        # Behaviour ------------------------
        df9 = df['behavioral_antiviral_meds_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df9['feature'] = 'Has taken antiviral medications'

        df10 = df['behavioral_avoidance_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df10['feature'] = 'Has avoided close contact with others with flu-like symptoms'

        df11 = df['behavioral_face_mask_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df11['feature'] = 'Has bought a face mask'

        df12 = df['behavioral_wash_hands_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df12['feature'] = 'Has frequently washed hands or used hand sanitizer'

        df13 = df['behavioral_large_gatherings_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df13['feature'] = 'Has reduced time at large gatherings'

        df14 = df['behavioral_outside_home_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df14['feature'] = 'Has reduced contact with people outside of own household'

        df15 = df['behavioral_touch_face_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df15['feature'] = 'Has avoided touching eyes, nose, or mouth'

        # Medical: Chronic condition and doctors recommendation
        df16 = df['doctor_recc_h1n1_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df16['feature'] = 'H1N1 flu vaccine'

        df17 = df['doctor_recc_seasonal_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df17['feature'] = 'Seasonal flu vaccine'

        # Features for private, personal
        df18 = df['chronic_med_condition_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df18['feature'] = 'Has chronic medical conditions'

        df19 = df['child_under_6_months_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df19['feature'] = 'Has regular close contact with child under 6 months'

        df20 = df['health_worker_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df20['feature'] = 'Is a healthcare worker'

        df21 = df['health_insurance_desc'].value_counts().rename_axis('value').reset_index(name='counts')
        df21['feature'] = 'Has health insurance'

        # Personal, family related features
        df22 = df['employment_status'].value_counts().rename_axis('value').reset_index(name='counts')
        df22['feature'] = 'employment_status'

        df23 = df['rent_or_own'].value_counts().rename_axis('value').reset_index(name='counts')
        df23['feature'] = 'rent_or_own'

        df24 = df['marital_status'].value_counts().rename_axis('value').reset_index(name='counts')
        df24['feature'] = 'marital_status'

        df25 = df['sex'].value_counts().rename_axis('value').reset_index(name='counts')
        df25['feature'] = 'sex'

        # Dataframe for personal features
        dfs_personal = [df22, df23, df24, df25]
        source_personal = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_personal)

        # Dataframe for health-related features
        dfs_health = [df18, df19, df20, df21]
        source_health = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_health)

        # Dataframe for medical features
        dfs_medical = [df16, df17]
        source_medical = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_medical)

        # Dataframes for h1n1 and seasonal flu
        dfs_h1n1 = [df3, df4, df5, df1, df2]
        dfs_seas = [df6, df7, df8]
        source_h1n1 = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_h1n1)
        source_seas = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_seas)

        # Dataframe for behavioural
        dfs_behavioral = [df9, df10, df11, df12, df13, df14, df15]
        source_behavioral = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_behavioral)

        # Population pyramid --------------------


        # Chart for personal features -----------
        # source_personal
        bars_personal = alt.Chart(source_personal, title='Personal features').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y(
                'feature:N', axis=alt.Axis(labelLimit=380)#,
                #sort=['Has health insurance',
                #        'Has chronic medical conditions',
                #        'Is a healthcare worker',
                #        'Has regular close contact with a child under the age of six months',
                #       ]
            ).title(''),
            color=alt.Color('value',
                            legend=alt.Legend(title=''),
                            scale=alt.Scale(scheme='rainbow'),
                            )
        ).configure_axis(
            labelFontSize=12,
            grid=False
        ).properties(
            width=500,
            height=250
        ).configure_view(
            strokeWidth=0
        )

        # Save chart as png file in dedicated folder
        bars_personal.save(FILE_BARCHART_FEATURES_PERSONAL)

        # Chart for health features -------------
        bars_health = alt.Chart(source_health, title='Other information about respondents').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y(
                'feature:N', axis=alt.Axis(labelLimit=380),
                sort=['Has health insurance',
                        'Has chronic medical conditions',
                        'Is a healthcare worker',
                        'Has regular close contact with a child under the age of six months',
                       ]
            ).title(''),
            color=alt.Color('value',
                            legend=alt.Legend(title=''),
                            scale=alt.Scale(scheme='rainbow'),
                            )
        ).configure_axis(
            labelFontSize=12,
            grid=False
        ).properties(
            width=500,
            height=250
        ).configure_view(
            strokeWidth=0
        )

        # Save chart as png file in dedicated folder
        bars_health.save(FILE_BARCHART_FEATURES_HEALTH)



        # Chart for medical features --------------
        bars_medical = alt.Chart(source_medical, title='Vaccine was recommended by doctor').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y(
                'feature:N', axis=alt.Axis(labelLimit=380),
                sort=['H1N1 flu vaccine',
                      'Seasonal flu vaccine']
            ).title(''),
            color=alt.Color('value',
                            legend=alt.Legend(title=''),
                            scale=alt.Scale(scheme='rainbow'),
                            )
        ).configure_axis(
            labelFontSize=12,
            grid=False
        ).properties(
            width=500,
            height=250
        ).configure_view(
            strokeWidth=0
        )

        # Save chart as png file in dedicated folder
        bars_medical.save(FILE_BARCHART_FEATURES_DOCTOR_REC)

        # Chart for behavioural --------------
        bars_behavioural = alt.Chart(source_behavioral, title='Behavioral').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y(
                'feature:N', axis=alt.Axis(labelLimit=380),
                sort=['Has frequently washed hands or used hand sanitizer',
                      'Has avoided close contact with others with flu-like symptoms',
                      'Has avoided touching eyes, nose, or mouth',
                      'Has reduced time at large gatherings',
                      'Has reduced contact with people outside of own household',
                      'Has bought a face mask',
                      'Has taken antiviral medications']
                    ).title(''),
            color=alt.Color('value',
                            legend=alt.Legend(title=''),
                            scale=alt.Scale(scheme='rainbow'),
                            )
        ).configure_axis(
            labelFontSize=12,
            grid=False
        ).properties(
            width=500,
            height=250
        ).configure_view(
            strokeWidth=0
        )

        # Save chart as png file in dedicated folder
        bars_behavioural.save(FILE_BARCHART_FEATURES_BEHAVIOURAL)

        # Chart for vaccines --------------
        # Chart for h1n1
        chart_h1n1 = alt.Chart(source_h1n1, title='Opinion on H1N1 flu vaccine').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y('feature:N').title(''),
            color = alt.Color('rating',
                              legend=alt.Legend(title="Ratings"),
                              scale=alt.Scale(scheme='lighttealblue')
                              )
            ).properties(
                width=250,
                height=250
            )

        # Chart for seasonal flu
        chart_seas = alt.Chart(source_seas, title='Opinion on seasonal vaccine').mark_bar().encode(
            x=alt.X('counts:Q').title(''),
            y=alt.Y('feature:N').title(''),
            color = alt.Color('rating')
            ).properties(
                width=250,
                height=150
            )

        # Combined chart
        chart = chart_h1n1 | chart_seas

        # Save chart as png file in dedicated folder
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

        bars = alt.Chart(df, title='Respondent received vaccines').mark_bar().encode(
            x=alt.X('count:Q').title('').stack('zero'),
            y=alt.Y('vaccine:N').title(''),
            color = alt.Color('vaccinated',
                          legend=alt.Legend(title=""),
                          scale=alt.Scale(scheme='lightgreyteal')
                          )
            ).properties(
                width=800,
                height=200
            ).configure_axis(
                labelFontSize=12,
                grid=False
            ).configure_view(
                strokeWidth=0
            )

        # Save the image in the fig folder
        bars.save(FILE_BARCHART_LABELS)


    def get_current_location(self):
        """
        Support function to return the current directly
        :return: prints the current directory
        """
        print(os.getcwd())