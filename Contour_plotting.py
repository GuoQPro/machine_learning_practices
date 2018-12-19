import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

abbr_data = pd.read_csv("Datasets/state-abbrevs.csv")
area_data = pd.read_csv("Datasets/state-areas.csv")
population_data = pd.read_csv("Datasets/state-population.csv")

print(abbr_data.head())
print(area_data.head())
print(population_data.head())

abbr_area = pd.merge(abbr_data, area_data)

print(abbr_area.head())
abbr_area_population = pd.merge(abbr_area, population_data, left_on = "abbreviation", right_on = "state/region")

abbr_area_population.drop(labels = ["state/region", "abbreviation"], axis = 1, inplace = True)
print(abbr_area_population.head())

def choose_total(x):
    return x["ages"] == "total"

annual_total_population = abbr_area_population.groupby(["state", "year", "ages"]).filter(choose_total)

print("Here is the count of population " % annual_total_population.groupby(["state", "year"]).sum())


alabama_annual_total_population = annual_total_population.loc[annual_total_population["state"] == "Alabama"]

print(alabama_annual_total_population)

sns.barplot(x = "year", y = "population", data = alabama_annual_total_population)
plt.show()
