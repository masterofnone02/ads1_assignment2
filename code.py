import pandas as pd
import matplotlib.pyplot as plt


def read_data(data_name):
    """
    This function reads the excel file and returns two dataframes.
    data_col_year = Returns dataframe with year as columns
    data_col_country = Returns dataframe with country as columns

    """
    data = pd.read_excel(data_name, header=None)
    df = data.iloc[4:]
    var = df.rename(columns=df.iloc[0]).drop(df.index[0])
    list_col = ['Country Code', 'Indicator Name', 'Indicator Code']
    var = var.drop(list_col, axis=1)
    df_year = var.set_index("Country Name")

    df_country = df_year.transpose()
    df_year.index.name = None
    df_country.index.name = None
    return df_year.fillna(0), df_country.fillna(0)

# def return_dataframes(data):

#     df_year_col = data[data["Country Name"].isin(countries)]
#     return df_year_col


def bar_graph(data_col, data_row, title=""):
    df = pd.DataFrame({'2000': data_col[2000.0],
                       '2010': data_col[2010.0],
                       '2020': data_col[2020.0]}, index=data_col.index)
   


countries = ["United States", "Senegal", "China", "Australia", "Germany", 
             "United Kingdom",
             "Japan", "Brazil"]

years = range(2000, 2021, 10)


ltr_year, ltr_coun = read_data("Labour_force_Ttl.xlsx")
ltr_wmn_year, ltr_wmn_coun = read_data("Labour_Force_Wmn.xlsx")

bar_graph(ltr_year, ltr_coun)
