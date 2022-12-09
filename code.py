import pandas as pd
import matplotlib.pyplot as plt


def read_data(data_name):
    """
    This function reads the excel file and returns two dataframes.
    data_col_year = Returns dataframe with year as columns
    data_col_country = Returns dataframe with country as columns

    """
    data = pd.read_excel(data_name, header=None)
    data = data.iloc[4:]
    var = data.rename(columns=data.iloc[0]).drop(data.index[0])
    list_col = ['Country Code', 'Indicator Name', 'Indicator Code']
    var = var.drop(list_col, axis=1)
    data_year = var.set_index("Country Name")

    data_country = data_year.transpose()
    data_year.index.name = None
    data_country.index.name = None
    return data_year.fillna(0), data_country.fillna(0)


def filter_dataframes(data):
    data = data[data.index.isin(countries)]
    return data


def bar_graph(data_col, data_row, title=""):
    data_col.columns.astype(int)
    df = pd.DataFrame({'1990': data_col[1990],
                       '2010': data_col[2010],
                       '2015': data_col[2015],
                       '2020': data_col[2020]}, index=data_col.index)
    df.plot.bar()
    plt.title(title)


def population_conversion(women, total, new_df):
    women.columns = women.columns.astype(int)
    for year in years:
        new_df[year] = (women[year]*total[year])/100
    return new_df


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


countries = ["United States", "Senegal", "China", "Australia", "Germany",
             "United Kingdom",
             "Japan", "Brazil"]

years = range(1990, 2021, 5)


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

mean_male_employment = pd.DataFrame({'Employment': ["Emplmnt in services",
                                                    "Emplmnt in Agriculture",
                                                    "Emplmnt in industries"],
                                     'Mean': [male_serv_year[2019.0].mean(),
                                              male_agr_year[2019.0].mean(),
                                              male_ind_year[2019.0].mean()]})

mean_female_employment = pd.DataFrame({'Employment': ["Emplmnt in services",
                                                      "Emplmnt in Agriculture",
                                                      "Emplmnt in industries"],
                                       'Mean': [fem_serv_year[2019.0].mean(),
                                                fem_agr_year[2019.0].mean(),
                                                fem_ind_year[2019.0].mean()]})


pie_chart(mean_male_employment, mean_female_employment)
