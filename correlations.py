from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_corr_plot(df):

    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


if __name__ == "__main__":
    data_dir = 'data/processed/'

    df = pd.read_csv(os.path.join(data_dir, 'response.csv'), index_col=0)

    f1 = 'IFNI'
    f2 = 'Q10.20'

    grouped = df.groupby(f1)
    for val, grp in grouped:
        print(grp[f2].value_counts() / len(grp))
