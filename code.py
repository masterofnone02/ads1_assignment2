import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_data(file):
    """
    This function returns two dataframes
    data_year = year as columns
    data_country = country as columns
    Parameters
    ----------
    file : String
        File name of the string to be read.

    Returns
    -------
    Dataframe
        Returns two dataframes with year as column and country as column.

    """
    data = pd.read_excel(file, header=None)
    data = data.iloc[4:]
    var = data.rename(columns=data.iloc[0]).drop(data.index[0])
    list_col = ['Country Code', 'Indicator Name', 'Indicator Code']
    var = var.drop(list_col, axis=1)
    data_year = var.set_index("Country Name")

    data_country = data_year.transpose()
    data_year.index.name = None
    data_country.index.name = None
    return data_year.fillna(0), data_country.fillna(0)


def filter_dataframes(data_frame):
    """
    This function filters the dataframe that is being passed based on country
    names present in the list countries

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame to be filtered based on countries.

    Returns
    -------
    data_frame : DataFrame
        Filtered DataFrame.

    """
    data_frame = data_frame[data_frame.index.isin(countries)]
    return data_frame


def filter_year_dataframe(data_frame):
    """
    This function filters the dataframe that is being passed based on years
    present in the list years

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame to be filtered based on year.

    Returns
    -------
    data_frame : DataFrame
        Filtered DataFrame.


    """
    data_frame = data_frame[data_frame.index.isin(years)]
    return data_frame


def bar_graph(col_df, row_df, title, xlbl, ylbl):
    """
    This function plots a bar graph with the two dataframe passed as argument with
    the title and labels.

    Parameters
    ----------
    col_df : DataFrame
        DataFrame for plotting bar graph with year as column.
    row_df : DataFrame
        DataFrame for plotting bar graph with country as column.
    title : String
        Title of the bar graph.
    xlbl : TYPE
        xlabel of the bar graph.
    ylbl : String
        ylabel of the bar graph.

    Returns
    -------
    None.

    """

    col_df.columns.astype(int)
    df = pd.DataFrame({'1990': col_df[1990],
                       '2010': col_df[2010],
                       '2015': col_df[2015],
                       '2020': col_df[2020]}, index=col_df.index)
    plt.figure()
    df.plot.bar()
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.savefig(title)
    plt.show()


def population_conversion(df, total, new_data_frame):
    """
    Converts the data of women labour as a percentage of total labour force

    Parameters
    ----------
    df : DataFrame
        DataFrame with percentage as value.
    total : DataFrame
        DataFrame with Total value to calculate percentage.
    new_data_frame : DataFrame
        DataFrame to store the generated values.

    Returns
    -------
    new_data_frame : DataFrame
        DataFrame with the percentage of women labour force.

    """
    df.columns = df.columns.astype(int)
    for year in years:
        new_data_frame[year] = (df[year]*total[year])/100
    return new_data_frame


def pie_chart(male, female):
    """
    Function to plot a pie chart with the dataframes passed

    Parameters
    ----------
    male : DataFrame
        Percentage of male in different service sectors.
    female : DataFrame
        Percentage of female in different service sectors..

    Returns
    -------
    None.

    """

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Male")
    plt.pie(male["Mean"])
    plt.subplot(1, 2, 2)
    plt.pie(female["Mean"])
    plt.title("Female")
    plt.suptitle("Male and Female employment in different sectors")
    plt.legend(labels=male["Employment"], bbox_to_anchor=[1, 1])
    plt.savefig("Male and Female employment in different sectors")
    plt.show()


def line_plot(data_frame, xlbl, ylbl):
    """
    Function to plot a line graph with the DataFrame passed along with labels.

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame to plot the line graph.
    xlbl : String
        xlabel for the graph.
    ylbl : String
        ylabel for the graph.

    Returns
    -------
    None.

    """

    df = pd.DataFrame({'China': data_frame["China"],
                       'Brazil': data_frame["Brazil"],
                       'United States': data_frame["United States"],
                       'Japan': data_frame["Japan"]}, index=data_frame.index)
    plt.figure()
    df.plot()
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title("Overall labour force of top countries over the year 1990-2020",
              fontsize=9)
    plt.savefig("Overall labour force of top countries over the year 1990-2020")
    plt.show()


def heatmap_df():
    """
    Function to generate the DataFrame for plotting the heatmap

    Returns
    -------
    DataFrame
        DataFrame with the indicators as index for China with years 1990, 2000
        and 2010.

    """

    df = pd.DataFrame({'indexes': indexes,
                       '1990': [labour_year[1990.0]["China"],
                                women_df[1990.0]["China"],
                                fem_ind_year[1990.0]["China"],
                                male_ind_year[1990.0]["China"],
                                fem_serv_year[1990.0]["China"],
                                male_serv_year[1990.0]["China"],
                                fem_serv_year[1990.0]["China"],
                                male_serv_year[1990.0]["China"]],
                       '2000': [labour_year[2000.0]["China"],
                                women_df[2000.0]["China"],
                                fem_ind_year[2000.0]["China"],
                                male_ind_year[2000.0]["China"],
                                fem_serv_year[2000.0]["China"],
                                male_serv_year[2000.0]["China"],
                                fem_serv_year[2000.0]["China"],
                                male_serv_year[2000.0]["China"]],
                       '2010': [labour_year[2010.0]["China"],
                                women_df[2010.0]["China"],
                                fem_ind_year[2010.0]["China"],
                                male_ind_year[2010.0]["China"],
                                fem_serv_year[2010.0]["China"],
                                male_serv_year[2010.0]["China"],
                                fem_serv_year[2010.0]["China"],
                                male_serv_year[2010.0]["China"]]}
                      )
    return df.set_index('indexes')


def heatmap(data_frame):
    """
    Function to plot heatmap with the DataFrame passed as argument.

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame generated using heatmap_df function.

    Returns
    -------
    None.

    """
    plt.figure()
    data_frame.index.name = None
    sns.heatmap(data_frame.transpose().corr(), cmap="Reds", annot=True)
    plt.title("China")
    plt.savefig("Heatmap for China")
    plt.show()


# List of countries to filter DataFrame
countries = ["United States", "Senegal", "China", "Yemen, Rep.", "Germany",
             "United Kingdom",
             "Japan", "Brazil"]

# Range of years
years = range(1990, 2021, 5)


# Index names used for heatmap generation
indexes = ["Total labour force", "Labour force wmn", "Employemnt in ind fem",
           "Employment in ind male", "Employment in serv fem",
           "Employment in serv male",
           "Employment in agr fem", "Employment in agr male"]

# Reading files and returning DataFrames
labour_year, labour_coun = read_data("Labour_force_Ttl.xlsx")
labour_wmn_year, labour_wmn_coun = read_data("Labour_Force_Wmn.xlsx")

fem_ind_year, femind_country = read_data("Employemnt_in_ind_fem.xls")
male_ind_year, maleind_count = read_data("Employment_in_ind_male.xls")

fem_serv_year, femserv_count = read_data("Employment_in_serv_fem.xls")
male_serv_year, maleserv_count = read_data("Employment_in_serv_male.xls")

fem_agr_year, femagr_count = read_data("Employment_in_agr_fem.xls")
male_agr_year, maleagr_count = read_data("Employment_in_agr_male.xls")

filtered_data_total = filter_dataframes(labour_year)
filtered_data_wmn = filter_dataframes(labour_wmn_year)

# Initialising an empty DataFrame
empty = pd.DataFrame()

# Returning DataFrame after population conversion
women_df = population_conversion(filtered_data_wmn, filtered_data_total, empty)


# Plotting bar graph
bar_graph(filtered_data_total, labour_coun, "Total Labour Force", "Countries",
          "Population")
bar_graph(women_df, labour_wmn_coun, "Labour Force Women", "Countries",
          "Population")

# Calculating mean of percentage for male employment in different sector using
# numpy

mean_male_emplymnt = pd.DataFrame({'Employment': ["Emplmnt in services",
                                                  "Emplmnt in Agriculture",
                                                  "Emplmnt in industries"],
                                   'Mean': [male_serv_year[2019.0].mean(),
                                            male_agr_year[2019.0].mean(),
                                            male_ind_year[2019.0].mean()]})
# Calculating mean of percentage for female employment in different sector
# using pandas dataframe
mean_female_emplymnt = pd.DataFrame({'Employment': ["Emplmnt in services",
                                                    "Emplmnt in Agriculture",
                                                    "Emplmnt in industries"],
                                     'Mean': [np.mean(fem_serv_year[2019.0]),
                                              np.mean(fem_agr_year[2019.0]),
                                              np.mean(fem_ind_year[2019.0])]})

# Plotting pie chart
pie_chart(mean_male_emplymnt, mean_female_emplymnt)
filtered_df = filter_year_dataframe(filtered_data_total.transpose())
line_plot(filtered_df, "Countries", "Years")
# Generating dataframe for plotting heatmap
heat_map = heatmap_df()
# Plotting heatmap
heatmap(heat_map)
