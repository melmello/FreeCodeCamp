import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where((df['weight'] / np.power((df["height"] / 100), 2)) > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df["cholesterol"] = np.where((df["cholesterol"] <= 1), 0, 1)
df["gluc"] = np.where((df["gluc"] <= 1), 0, 1)


def draw_cat_plot():
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.

    # Draw the catplot with 'sns.catplot()'
    df_cat = pd.melt(df[['cardio', 'active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']],
                     id_vars=['cardio'],
                     var_name='variable',
                     value_name='value')

    fig = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind='count')
    fig = fig.set_axis_labels("variable", "total").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) &
                     (df['height'] >= df['height'].quantile(0.025)) &
                     (df['height'] <= df['height'].quantile(0.975)) &
                     (df['weight'] >= df['weight'].quantile(0.025)) &
                     (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix

    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(15, 10))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, mask=mask, square=True, annot=True, fmt=".1f")

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
