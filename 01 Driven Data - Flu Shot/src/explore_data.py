import pandas as pd
import os
import altair as alt
import functools
import seaborn as sns
import numpy as np

# TODO : log data to file not print to screen

FILENAME_INPUT_DATA_LABELS = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv'
FILENAME_INPUT_DATA_FEATURES = '../data/Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv'

FILE_BARCHART_LABELS = '../fig/labels_bar_chart.png'
FILE_BARCHART_FEATURES_SENTIMENT = '../fig/features_bar_chart_sentiment.png'
FILE_BARCHART_FEATURES_BEHAVIOURAL = '../fig/features_bar_chart_behavioral.png'
FILE_BARCHART_FEATURES_DOCTOR_REC = '../fig/features_bar_chart_doctor_recommendation.png'
FILE_BARCHART_FEATURES_HEALTH = '../fig/features_bar_chart_health.png'
FILE_BARCHART_FEATURES_PERSONAL = '../fig/features_bar_chart_personal.png'
FILE_BARCHART_FEATURES_EMPLOYMENT = '../fig/features_bar_chart_employment.png'
FILE_BARCHART_FEATURES_POPULATION = '../fig/features_bar_chart_population_pyramid.png'
FILE_BARCHART_FEATURES_HOUSEHOLD = '../fig/features_bar_chart_household.png'

class RawFluShotData:
    """
    Class responsible for the exploration of Flu Shot Data
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

        # print(columns_explore)
        # print(df_train.columns)

        # print(f'Dataset has {df_train.shape[0]} entries.')
        # print(f'Dataset has columns : {df_train.columns.tolist()}.')

        # from Driven Data
        # Let's double-check that the rows between the features and the labels match up.
        # We don't want to have the wrong labels.
        # Numpy's assert_array_equal will error if the two arrays—the row indices of the two data frames
        # don't match up.
        # np.testing.assert_array_equal(features_df.index.values, labels_df.index.values)

        # Feature exploration
        # 1. Features with rating -----------------------------------
        # Create a new feature (str) with description and a new int feature
        # Features with description to be used for visual exploration,
        # Features with int values to be used for modeling.

        # Different features might have different ratings / scales
        # We translate sentiments to the same scale

        # H1N1_concern
        df_train['h1n1_concern_desc'] = df_train['h1n1_concern'].map({
                    0.0: 'None',
                    1.0: 'A little',
                    2.0: 'Somewhat',
                    3.0: 'A lot'
                    }
                )

        # H1N1_knowledge
        df_train['h1n1_knowledge_desc'] = df_train['h1n1_knowledge'].map({0.0: 'None',
                       1.0: 'A little',
                       2.0: 'A lot'})

        # Scale for effective, risk and sick
        map_effective_risk_sick_desc = {1.0: 'None',
                                        2.0: 'A little',
                                        3.0: 'Dont know',
                                        4.0: 'Somewhat',
                                        5.0: 'A lot'}

        # Apply the map
        for c in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                  'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc']:
            # Replace numeric values by description
            df_train[f'{c}_desc'] = df_train[c].map(map_effective_risk_sick_desc)

        # Replace missing values NaN by 'dont know'
        for c in ['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                  'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc',
                  'h1n1_concern', 'h1n1_knowledge']:
            df_train[f'{c}_desc'] = df_train[f'{c}_desc'].fillna('Dont know')

        # 2. Behavioral features Y/N
        # Create a new feature (str) with Y for 1, N for 0 and 'No response' for NaN
        # We address missing values in responses for behavioral features adding 'dont know' option.

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

        # Apply the mapping function for behavioral features and return stats
        for f in features_behavioral:
            df_train[f'{f}_desc'] = df_train[f].apply(float_to_word)
            # print output to log file
            # print(100*df_train[f'{f}_int'].value_counts()/df_train.shape[0])
            # print('-'*10)

        # -------------------------------------------------------
        # 3. Chronic condition and doctors recommendation
        # Personal
        features_medical = ['doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition']
        features_health = ['chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance']

        for f in features_medical + features_health:
            df_train[f'{f}_desc'] = df_train[f].apply(float_to_word)

        # 4. Explore other those and other features plotting charts
        self.plot_stacked_bar_behaviour_medical_personal(df_train)
        self.plot_diverging_stacked_bar(df_train)
        self.plot_bar_random_namings(df_train)
        self.plot_population_pyramid(df_train)

        # -------------------------------------------------------
        # 5. Explore labels as is, without features
        columns = ['h1n1_vaccine', 'seasonal_vaccine']
        self.explore_labels(df_train[columns])

    def plot_diverging_stacked_bar(self, df):
        """
        Plot diverging stacked bar chart for sentiments towards a set of questions,
        displayed as percentages with neutral responses straddling the 0% mark.
        :return: Chart saved as a png file.
        """

        features_sentiment = ['opinion_h1n1_vacc_effective_desc', 'opinion_h1n1_risk_desc',
                              'opinion_h1n1_sick_from_vacc_desc',  'opinion_seas_vacc_effective_desc',
                              'opinion_seas_risk_desc', 'opinion_seas_sick_from_vacc_desc',
                              'h1n1_concern_desc', 'h1n1_knowledge_desc']

        df1 = df['opinion_h1n1_vacc_effective_desc'].value_counts().rename_axis('type').reset_index(name='value')
        df2 = df['opinion_h1n1_risk_desc'].value_counts().rename_axis('type').reset_index(name='value')

        df3 = df['opinion_h1n1_sick_from_vacc_desc'].value_counts().rename_axis('type').reset_index(name='value')
        df4 = df['opinion_seas_vacc_effective_desc'].value_counts().rename_axis('type').reset_index(name='value')

        df5 = df['opinion_seas_risk_desc'].value_counts().rename_axis('type').reset_index(name='value')
        df6 = df['opinion_seas_sick_from_vacc_desc'].value_counts().rename_axis('type').reset_index(name='value')

        df7 = df['h1n1_concern_desc'].value_counts().rename_axis('type').reset_index(name='value')
        df8 = df['h1n1_knowledge_desc'].value_counts().rename_axis('type').reset_index(name='value')

        df1['question'] = "Respondent's opinion about H1N1 vaccine effectiveness"
        df2['question'] = "Respondent's opinion about risk of getting sick with H1N1 flu without vaccine"
        df3['question'] = "Respondent's worry of getting sick from taking H1N1 vaccine"
        df4['question'] = "Respondent's opinion about seasonal flu vaccine effectiveness"
        df5['question'] = "Respondent's opinion about risk of getting sick with seasonal flu without vaccine"
        df6['question'] = "Respondent's worry of getting sick from taking seasonal flu vaccine"
        df7['question'] = "Level of concern about the H1N1 flu"
        df8['question'] = "Level of knowledge about H1N1 flu"

        # Dataframe for medical features
        dfs_sentiment = [df1, df2, df3, df4, df5, df6, df7, df8]
        source_sentiment = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_sentiment)

        # Add type_code that we can sort by
        source_sentiment['type_code'] = source_sentiment['type'].map(
            {
                "None": -2,
                "A little": -1,
                "Dont know": 0,
                "Somewhat": 1,
                "A lot": 2,
            }
        )

        def compute_percentages(
                group,
        ):
            # Set type_code as index and sort
            group = group.set_index("type_code").sort_index()

            # Compute percentage of value with question group
            perc = (group["value"] / group["value"].sum()) * 100
            group["percentage"] = perc

            # Compute percentage end, centered on "Neither agree nor disagree" (type_code 0)
            # Note that we access the perc series via index which is based on 'type_code'.
            group["percentage_end"] = perc.cumsum() - (perc[-2] + perc[-1] + perc[0] / 2)

            # Compute percentage start by subtracting percent
            group["percentage_start"] = group["percentage_end"] - perc

            return group

        source_sentiment  = source_sentiment.groupby("question").apply(compute_percentages).reset_index(drop=True)

        color_scale = alt.Scale(
            domain=[
                "None",
                "A little",
                "Dont know",
                "Somewhat",
                "A lot",
            ],
            range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab"],
        )

        y_axis = alt.Axis(title="", offset=5, ticks=False, minExtent=60, domain=False, labelLimit=380)

        bar_chart = alt.Chart(source_sentiment).mark_bar().encode(
            x=alt.X("percentage_start:Q"),
            x2="percentage_end:Q",
            y=alt.Y("question:N").axis(y_axis).sort('-x'),
            color=alt.Color("type:N").title("Response").scale(color_scale)
        )

        bar_chart.save(FILE_BARCHART_FEATURES_SENTIMENT)

    def plot_population_pyramid(self, df):
        """
        Create population pyramid for respondents
        :param df: raw train data
        :return: chart for population pyramid
        """

        # Group data by age and sex, count all instances
        df = df.groupby(['age_group', 'sex'])['respondent_id'].count().reset_index()
        df = df.rename(columns={'respondent_id': 'counts', 'sex':'gender', 'age_group':'age'})

        # Plot chart
        base = alt.Chart(df).properties(
            width=250
        )

        color_scale = alt.Scale(domain=['Male', 'Female'],
             range=['#1f77b4', '#e377c2'])

        # Left component, female
        left = base.transform_filter(
            alt.datum.gender == 'Female'
        ).encode(
            alt.Y('age:O').axis(None),
            alt.X('counts:Q')
            .title('population')
            .sort('descending'),
            alt.Color('gender:N')
            .scale(color_scale)
            .legend(None)
        ).mark_bar().properties(title='Female')

        # Middle component, age groups
        middle = base.encode(
            alt.Y('age').axis(None),
            alt.Text('age'),
        ).mark_text().properties(width=20)

        # Right component, male
        right = base.transform_filter(
            alt.datum.gender == 'Male'
        ).encode(
            alt.Y('age:O').axis(None),
            alt.X('counts:Q').title('population'),
            alt.Color('gender:N').scale(color_scale).legend(None)
        ).mark_bar().properties(title='Male')

        # Combined chart
        chart = alt.concat(left, middle, right, spacing=5)

        chart.save(FILE_BARCHART_FEATURES_POPULATION)


    def plot_bar_random_namings(self, df):
        """
        Explore employment-related features
        :param df: df train
        :return: Chart in png files
        """

        df_employment_occupation = df['employment_occupation'].value_counts().rename_axis('value').reset_index(name='counts')
        df_employment_industry = df['employment_industry'].value_counts().rename_axis('value').reset_index(name='counts')

        def plot_inner_chart(source, title):
            """
            Support function for plotting
            :param source: source for data
            :param title: title for the chart
            :return: chart with base and text
            """
            base = alt.Chart(source, title=title).encode(
                x=alt.X('counts').title(''),
                y=alt.Y("value").sort('-x').title(''),
                text='counts',
                color=alt.condition(alt.datum.counts > 1000, alt.value('red'), alt.value('steelblue'))
            )
            chart_ = base.mark_bar() + base.mark_text(align='left', dx=2)

            return chart_

        chart_occupation = plot_inner_chart(df_employment_occupation, 'Type of occupation of respondent')
        chart_industry = plot_inner_chart(df_employment_industry, 'Type of industry respondent is employed in')

        chart_employment = chart_occupation | chart_industry

        chart_employment.save(FILE_BARCHART_FEATURES_EMPLOYMENT)

    def plot_stacked_bar_behaviour_medical_personal(self, df):
        """
        Plot graphs to explore features.
        :param df: train dataframe with all features
        :return: bar chart saved as png file
        """

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


        # Dataframe for health-related features
        dfs_health = [df18, df19, df20, df21]
        source_health = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_health)

        # Dataframe for medical features
        dfs_medical = [df16, df17]
        source_medical = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_medical)

        # Dataframe for behavioural
        dfs_behavioral = [df9, df10, df11, df12, df13, df14, df15]
        source_behavioral = functools.reduce(lambda left, right: pd.concat([left, right]), dfs_behavioral)

        def plot_inner_chart(feature):
            """
            Creates a chart per feature
            :param feature: feature (column) in scope
            :return: sub plot for the overall chart
            """

            bar_ = alt.Chart(df).mark_bar().encode(
                x=alt.X(f'{feature}:O').title(''),
                y=alt.Y(f'count({feature}):Q', scale=alt.Scale(domain=[0, 19000])).title(''),
                color = alt.Color(f'{feature}:N',
                                  legend=None)
            )
            return bar_

        bar_sex = plot_inner_chart('sex')
        bar_census_msa = plot_inner_chart('census_msa')
        bar_empl_status = plot_inner_chart('employment_status')
        bar_rent_or_own = plot_inner_chart('rent_or_own')
        bar_marital_status = plot_inner_chart('marital_status')
        bar_education = plot_inner_chart('education')
        bar_race = plot_inner_chart('race')
        bar_income = plot_inner_chart('income_poverty')

        chart = bar_sex | bar_empl_status | bar_census_msa | bar_rent_or_own | bar_marital_status | bar_education | bar_race | bar_income
        chart.save(FILE_BARCHART_FEATURES_PERSONAL)

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
        bars_behavioural = alt.Chart(source_behavioral).mark_bar().encode(
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
            color=alt.Color("value:N").title("Response")
        ).configure_axis(
            labelFontSize=12,
            grid=False
        ).properties(
            width=500,
            height=250
        ).configure_view(
            strokeWidth=0
        )

        bars_behavioural.save(FILE_BARCHART_FEATURES_BEHAVIOURAL)

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

        #
        # print(df_counts_percentages)

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