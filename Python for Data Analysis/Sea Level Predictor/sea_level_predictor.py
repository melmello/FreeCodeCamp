import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


def draw_plot():
    # Read data from file
    df = pd.read_csv("epa-sea-level.csv", float_precision="legacy")

    # Create scatter plot
    x = df["Year"]
    y = df["CSIRO Adjusted Sea Level"]

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(x, y)

    # Create first line of best fit
    first_year = df["Year"].loc[df["Year"].idxmin()]
    res1 = linregress(x, y)
    # using 2051 because of new pandas version
    ax.plot(range(first_year, 2051), res1.intercept + res1.slope * range(df["Year"].min(), 2051), "r")

    # Create second line of best fit
    res2 = linregress(df.loc[(df.Year >= 2000), "Year"].values,
                      df.loc[(df.Year >= 2000), "CSIRO Adjusted Sea Level"].values)
    # using 2051 because of new pandas version
    ax.plot(range(2000, 2051), res2.intercept + res2.slope * range(2000, 2051), "g")

    # Add labels and title
    ax.set_title("Rise in Sea Level")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (inches)")

    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig("sea_level_plot.png")
    return plt.gca()
