import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to "date".)
df = pd.read_csv("fcc-forum-pageviews.csv", parse_dates=["date"], index_col="date")

# Clean data
df = df.loc[(df["value"] >= df["value"].quantile(0.025)) & (df["value"] <= df["value"].quantile(0.975))]


def draw_line_plot():
    # Draw line plot

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(30, 15), dpi=100)

    ax = sns.lineplot(x="date", y="value", data=df)
    ax.set_title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019")
    ax.set_xlabel("Date")
    ax.set_ylabel("Page Views")

    # Save image and return fig (don"t change this part)
    fig.savefig("line_plot.png")
    return fig


def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df.copy()
    df_bar["year"] = df.index.year
    df_bar["month"] = df.index.month_name()

    df_bar = pd.DataFrame(df_bar.groupby(["year", "month"], sort=False)["value"].mean().round().astype(int))
    df_bar = df_bar.reset_index()

    hue_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(15, 10))
    # Draw bar plot
    ax = sns.barplot(x="year", y="value", hue="month", hue_order=hue_order, data=df_bar, palette="tab10")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment="center")

    ax.set_title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019")
    ax.set_xlabel("Years")
    ax.set_ylabel("Average Page Views")

    # Save image and return fig (don"t change this part)
    fig.savefig("bar_plot.png")
    return fig


def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box["year"] = [d.year for d in df_box.date]
    df_box["month"] = [d.strftime("%b") for d in df_box.date]

    # Draw box plots (using Seaborn)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1, 2, figsize=(32, 10))

    ax[0] = sns.boxplot(x="year", y="value", data=df_box, ax=ax[0])
    ax[0].set_title("Year-wise Box Plot (Trend)")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Page Views")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax[1] = sns.boxplot(x="month", y="value", order=months, data=df_box, ax=ax[1])
    ax[1].set_title("Month-wise Box Plot (Seasonality)")
    ax[1].set_xlabel("Month")
    ax[1].set_ylabel("Page Views")

    # Save image and return fig (don"t change this part)
    fig.savefig("box_plot.png")
    return fig
