#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:43:06 2023

@author: tayssirboukrouba
"""

# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import stats as st
import numpy as np
import seaborn as sns
import warnings

# Suppress a specific warning
warnings.filterwarnings("ignore")

# defining the functions

def read_and_transpose(filename):
    '''
    Reads a filename of csv dataframe and returns 2 dataframes

            Parameters:
                   filename (String): a string of filename source 

            Returns:
                   year_df (DataFrame) : containg years as columns
                   country_df (DataFrame) : containg countries as columns
    '''
    # reading the data
    df = pd.read_csv(filename, na_values='..')

    # cleaning the data
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['Series Code', 'Country Code'], inplace=True)

    year_df = df.set_index(['Series Name', 'Country Name'])
    year_df.columns = [str(year) for year in range(2010, 2021)]

    # creating the countries dataframe
    country_df = pd.DataFrame.transpose(df)
    header = country_df.iloc[1].values.tolist()
    country_df.columns = header
    country_df = country_df.iloc[2:]
    country_df = country_df.apply(pd.to_numeric, errors='coerce')


    # returning the dataframes
    return year_df, country_df


def lineplot(df, indicator, countries, title, xlabel, ylabel):
    '''
    Creates a line plot of selected countries and indicator values over x years

            Parameters:
                   df (DataFrame): dataframe containing the columns
                   indicator (String): indiator column name 
                   countries(list): list of countries to be selected 
                   title (String) : title of the plot
                   x_label (String) : label on x axis
                   y_label (String) : label on y axis

            Returns:
                    None
    '''
    df = df.loc[indicator]
    years = df.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 7))

    for country in countries:
        ax.plot(years, df.loc[country], label=country)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axvline(x='2019', color='black', linestyle='--', label='Pendamic')
    plt.legend()
    custom_labels = range(2010, 2021)
    ax.set_xticklabels(custom_labels)
    plt.grid(axis='both', alpha=.3)
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.style.use('seaborn-whitegrid')


def correlation_mat(df, country, title):
    '''
    Creates a correlation matrix for selected country.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        country (String): country to be selected.

    Returns:
        None.
    '''

    condition = df.index.get_level_values('Country Name') == country
    result_df = df.loc[condition].reset_index(level=1, drop=True)
    correlation_matrix = result_df.T.corr()
    col_rename = ['Rural pop', 'Urban Pop', 'Electricity Access',
                  'Internet Usage', 'Secure Servers',
                  'Mobile cellular subs', 'Phone subs',
                  'Broadland subs',
                  'GDP', 'ICT Exports']
    correlation_matrix.columns = col_rename
    correlation_matrix.index = col_rename
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True,
                cmap='BuPu', fmt=".2f", linewidths=.5)
    plt.xticks(ticks=np.arange(0.5, len(col_rename)))
    plt.title(title)


def barplot(df, indicator, years, countries, xlabel, ylabel, title):
    '''
    Creates a bar plot for selected countries, displaying indicator values over years.

    Parameters:
        df (DataFrame): DataFrame containing the columns.
        indicator (String): Indicator column name.
        years (list): List of years to be selected.
        countries (list): List of countries to be selected.
        xlabel (String): Label on the x-axis.
        ylabel (String): Label on the y-axis.
        title (String): Title of the plot.

    Returns:
        None.
    '''
    country_filter = df.index.get_level_values('Country Name').isin(countries)
    indicator_filter = df.index.get_level_values('Series Name') == indicator

    df = df[years].loc[(indicator_filter) & (country_filter)]
    plt.figure()
    df.reset_index(level='Series Name', inplace=True)
    plottype = 'bar'
    colormap = 'viridis'
    df.plot(kind=plottype, width=0.8, figsize=(
        8, 6), rot=0, cmap=colormap, edgecolor='k')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def pyramid_plot(df, indicator1, indicator2, country, years, label1, label2, title):
    '''
    Creates a pyramid plot for a specific country, comparing two indicators over years.

    Parameters:
        df (DataFrame): DataFrame containing the columns.
        indicator1 (String): First indicator column name.
        indicator2 (String): Second indicator column name.
        country (String): Country to be selected.
        years (list): List of years to be selected.
        label1 (String): Label for the first indicator.
        label2 (String): Label for the second indicator.
        title (String): Title of the plot.

    Returns:
        None.
    '''

    fig, ax = plt.subplots(figsize=(15, 10))
    data1 = df.loc[(indicator1, country)]
    data2 = df.loc[(indicator2, country)]

    # Plot the left side of the pyramid
    ax.barh(years, data1, color='steelblue',
            edgecolor='black', height=0.6, label=label1)

    # Plot the right side of the pyramid
    ax.barh(years, [-value for value in data2], color='indianred',
            edgecolor='black', height=0.6, label=label2)

    # Remove y-axis ticks
    ax.tick_params(axis='y', which='both', left=False, right=False)

    # Set labels and title
    ax.set_xlabel('Value')
    ax.set_title(title)
    plt.legend()

    # Display the plot
    plt.show()


def scatterplot(df, countries, indicator1, indicator2, title):
    '''
    Creates a scatter plot comparing two indicators for selected countries.

    Parameters:
        df (DataFrame): DataFrame containing the columns.
        countries (list): List of countries to be selected.
        indicator1 (String): First indicator column name.
        indicator2 (String): Second indicator column name.

    Returns:
        None.
    '''
    for country in countries:
        x = df.loc[(indicator1, country)]
        y = df.loc[(indicator2, country)]
        plt.scatter(x=np.log10(x), y=np.log10(y), s=100, label=country)
        plt.legend()
    plt.xlabel(indicator1)
    plt.ylabel(indicator2)
    plt.title(title)


# reading the csv file
filename = '7aca94a2-d49e-4bd1-a398-0044d694f975_Data.csv'
yrdf, cdf = read_and_transpose(filename)


# EDA using decribe
print('Summary Stats of year_df :', yrdf.describe(), sep='\n')
print('Summary Stats of country_df :', cdf.describe(), sep='\n')

# EDA using Kurtosis and Skewness
print('kurtosis (Morroco) :', st.kurtosis(cdf['Morocco']), sep='\n')
print('Skewness (Morroco) :', st.skew(cdf['Morocco']), sep='\n')

print('kurtosis (Peru) :', st.kurtosis(cdf['Peru']), sep='\n')
print('Skewness (Peru) :', st.skew(cdf['Peru']), sep='\n')

# calculating correlation
col1 = 'Urban population'
col2 = 'Fixed broadband subscriptions'
country1 = 'Tanzania'
country2 = 'Bolivia'

print('Correlations :')
print(yrdf.loc[(col1, country1)].corr(yrdf.loc[(col2, country1)]))
print(yrdf.loc[(col1, country2)].corr(yrdf.loc[(col2, country2)]))

# defining variables for pyramid plots :
ind1 = 'Urban population'
ind2 = 'Rural population'
years = yrdf.columns
title1 = 'Population Perportions in South America'
title2 = 'Population Perportions in Central/Lower Africa'
title3 = 'Population Perportions in North Africa'

# calling pyramid_plot() function :
pyramid_plot(yrdf, ind1, ind2, 'Bolivia', years, ind1, ind2, title1)
pyramid_plot(yrdf, ind1, ind2, 'Ghana', years, ind1, ind2, title2)
pyramid_plot(yrdf, ind1, ind2, 'Algeria', years, ind1, ind2, title3)


# defining variables for lineplots :
ind1 = 'ICT service exports (BoP, current US$)'
ind2 = 'Individuals using the Internet (% of population)'
title1 = 'ICT Service Exports of African And South American Continent'
title2 = 'Internet Usage in African And South American Continent'
countries = ['Algeria', 'Kenya', 'Morocco',
             'Ghana', 'Tanzania', 'Bolivia', 'Peru']

# calling lineplot() function :
lineplot(yrdf, ind1, countries, title1, 'years', ind1)
lineplot(yrdf, ind2, countries, title2, 'years', ind2)

# defining variables for heatmaps :
title1 = 'Correlation Matrix between Indicators of Kenya'
title2 = 'Correlation Matrix between Indicators of Peru'

# calling correlation_mat() function :
correlation_mat(yrdf, 'Kenya', title1)
correlation_mat(yrdf, 'Peru', title2)

# defining variables for barplots :
ind1 = 'Secure Internet servers'
ind2 = 'Fixed telephone subscriptions'
title1 = 'Internet Servers Security in Africa and South America'
title2 = 'Fixed telephone subscriptions in Africa and America'
years = ['2010', '2012', '2014', '2016', '2018', '2020']
countries = ['Algeria', 'Kenya', 'Morocco',
             'Ghana', 'Tanzania', 'Bolivia', 'Peru']


# calling barplot() function :
barplot(yrdf, ind1, years, countries, 'countries', ind1, title1)
barplot(yrdf, ind2, years, countries, 'countries', ind2, title2)


# defining variable for scatterplots :
countries = ['Algeria', 'Tanzania', 'Peru']
ind1 = 'Fixed broadband subscriptions'
ind2 = 'Urban population'
ind3 = 'Mobile cellular subscriptions'
ind4 = 'Rural population'
title1 = 'Relationship between Urban Pop and Broadband subs across Countries'
title2 = 'Relation between Rural Pop and Mobile cellular subs across Countries'

# calling scatterplot() function :
plt.figure(figsize=(10, 7))
scatterplot(yrdf, countries, ind1, ind2, title1)
plt.figure(figsize=(10, 7))
scatterplot(yrdf, countries, ind3, ind4, title2)
