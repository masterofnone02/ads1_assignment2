import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_data(file):
    """
    This function reads the excel file and returns two dataframes.
    col_df_year = Returns dataframe with year as columns
    col_df_country = Returns dataframe with country as columns

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
    data_frame = data_frame[data_frame.index.isin(countries)]
    return data_frame


def filter_year_dataframe(data_frame):
    data_frame = data_frame[data_frame.index.isin(years)]
    return data_frame


def bar_graph(col_df, row_df, title=""):
    col_df.columns.astype(int)
    df = pd.DataFrame({'1990': col_df[1990],
                       '2010': col_df[2010],
                       '2015': col_df[2015],
                       '2020': col_df[2020]}, index=col_df.index)
    df.plot.bar()
    plt.title(title)


def population_conversion(women, total, new_data_frame):
    women.columns = women.columns.astype(int)
    for year in years:
        new_data_frame[year] = (women[year]*total[year])/100
    return new_data_frame


def pie_chart(male, female):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Male")
    plt.pie(male["Mean"])
    plt.subplot(1, 2, 2)
    plt.pie(female["Mean"])
    plt.title("Female")
    plt.suptitle("Male and Female employment in different sectors")
    plt.legend(labels=male["Employment"], bbox_to_anchor=[1, 1])
    plt.show()


def line_plot(data_frame):

    df = pd.DataFrame({'China': data_frame["China"],
                       'Brazil': data_frame["Brazil"],
                       'United States': data_frame["United States"],
                       'Japan': data_frame["Japan"]}, index=data_frame.index)
    df.plot()
    plt.title("Overall labour force of top countries over the year 1990-2020",
              fontsize=9)


def heatmap_df():

    labour_year, labour_coun = read_data("Labour_force_Ttl.xlsx")
    labour_wmn_year, labour_wmn_coun = read_data("Labour_Force_Wmn.xlsx")

    fem_ind_year, femind_country = read_data("Employemnt_in_ind_fem.xls")
    male_ind_year, maleind_count = read_data("Employment_in_ind_male.xls")

    fem_serv_year, femserv_count = read_data("Employment_in_serv_fem.xls")
    male_serv_year, maleserv_count = read_data("Employment_in_serv_male.xls")

    fem_agr_year, femagr_count = read_data("Employment_in_agr_fem.xls")
    male_agr_year, maleagr_count = read_data("Employment_in_agr_male.xls")

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
    plt.figure()
    data_frame.index.name = None
    sns.heatmap(data_frame.transpose().corr(), cmap="Reds", annot=True)
    plt.title("")
    plt.show()


countries = ["United States", "Senegal", "China", "Australia", "Germany",
             "United Kingdom",
             "Japan", "Brazil"]

years = range(1990, 2021, 5)


indexes = ["Total labour force", "Labour force wmn", "Employemnt in ind fem",
           "Employment in ind male", "Employment in serv fem",
           "Employment in serv male",
           "Employment in agr fem", "Employment in agr male"]


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

empty = pd.DataFrame()
women_df = population_conversion(filtered_data_wmn, filtered_data_total, empty)


bar_graph(filtered_data_total, labour_coun, title="Total Labour Force")
bar_graph(women_df, labour_wmn_coun, title="Labour Force Women")

mean_male_emplymnt = pd.DataFrame({'Employment': ["Emplmnt in services",
                                                    "Emplmnt in Agriculture",
                                                    "Emplmnt in industries"],
                                     'Mean': [np.mean(male_serv_year[2019.0]),
                                              np.mean(male_agr_year[2019.0]),
                                              np.mean(male_ind_year[2019.0])]})

mean_female_emplymnt = pd.DataFrame({'Employment': ["Emplmnt in services",
                                                     "Emplmnt in Agriculture",
                                                     "Emplmnt in industries"],
                                     'Mean': [np.mean(fem_serv_year[2019.0]),
                                              np.mean(fem_agr_year[2019.0]),
                                              np.mean(fem_ind_year[2019.0])]})

pie_chart(mean_male_emplymnt, mean_female_emplymnt)
filtered_df = filter_year_dataframe(filtered_data_total.transpose())
line_plot(filtered_df)
a = heatmap_df()
heatmap(a)
