# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: 'Python 3.10.0 64-bit (''cmpe-351-_rWzjxJw'': pipenv)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMPE 351 Exercise 1
#
# I'm choosing to get familiar with using VS Code for running Python and R for
# data exploration. I'm deciding to stick with using just Python for this
# exercise, using `pipenv` to manage my packages and virtual environments.

# %%
# Importing packages.
from os.path import dirname, join, realpath

# Data collection and processing
import pandas as pd

# Data visualization
import seaborn as sns
from scipy.stats import spearmanr

# %%
# Reading the data file relative to the current file, based on whther or not
# we're in a jupyter notebook or a script file, as I tend to edit code in
# script files to take advantage of linting and formatting.


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    script_dir = dirname(realpath("__file__"))
else:
    script_dir = dirname(realpath(__file__))

raw_listings_df = pd.read_csv(join(script_dir, "data/listings.csv"))

# %%
raw_listings_df.head()

# %%
print(raw_listings_df.info())

# %% [markdown]
# ## RQ1 (30 points)
#
# > Provide at least three types of statistical summaries of the listing price
# > ("price" column) in the data set. You can use plot or numeric values. Write
# > a summary in the markdown cell, stating your finding(s) from your provided
# > statistics summaries.

# %%
pricing_series = raw_listings_df["price"]

# %% [markdown]
# First, let's look at the Five-number summary to see the [markdown]
# distribution of the listing prices.

# %%
print(pricing_series.describe())

# %% [markdown]
# It's clear that there are large outliers when looking that the [markdown]
# 3rd quartile and the maximum value. Let's look at a boxplot of the listing
# prices to confirm this observation.

# %%
# Boxplot
sns.boxplot(x=pricing_series)

# %% [markdown]
# The abundant high value outliers make it difficult to observe [markdown]
# the main part of the boxplot. Let's not show the outliers in another boxplot.

# %%
# Without the outliers
sns.boxplot(x=pricing_series, showfliers=False)

# %% [markdown]
# With a better idea of the outliers and the location of the data, let's now look at how spread out the data is and what shape it takes.
#
# For the sake of visualization, we'll filter out the large valued outliers first.

# %%
# Source: https://datascience.stackexchange.com/a/57199/131303
first_quantile = pricing_series.quantile(0.25)
third_quantile = pricing_series.quantile(0.75)
interquantile_range = third_quantile - first_quantile
filter = (pricing_series > first_quantile - 1.5 * interquantile_range) & (
    pricing_series < third_quantile + 1.5 * interquantile_range
)

filtered_pricing_series = pricing_series[filter]

# %%
# Histogram
sns.histplot(filtered_pricing_series, binwidth=5)

# %%
# Density plot
sns.kdeplot(filtered_pricing_series)

# %% [markdown]
# ### Summary
#
# The listing price data is spread out over a range of $0-$250, but has numerous outliers ranging from $250-$13 300. The shape of the listing price data is roughly a normal distribution that's slightly skewed to the right.

# %% [markdown]
# ## RQ2 (20 points)
#
# > Design and perform a statistical test involving list room
# > type and list price. In a markdown cell, describe the null hypothesis
# > statement of your test and why you choose a specific statistical test method.
# > You can consider more attributes if you want.
#
# Null hypothesis: Listing room type has no effect on listing price.
# Alternative hypothesis: Listing room type has an effect on listing price.
#
# Statistical test method: Spearman's Rank Correlation.
# Reasoning: I want to test the relationship between a listing's room type and its price. Since there's no independent variable, and I can't assume that each room type has normally distributed prices, I decided to go with a non-parametric test.

# %%
room_type_price_df = raw_listings_df[["room_type", "price"]][filter]
descending_median_order = (
    room_type_price_df.groupby(["room_type"])["price"]
    .median()
    .sort_values(ascending=False)
    .index
)

sns.violinplot(
    x="room_type",
    y="price",
    data=room_type_price_df,
    order=descending_median_order,
)

# %%
stat, p = spearmanr(
    room_type_price_df["room_type"], room_type_price_df["price"]
)

print("stat=%.3f, p=%.3f" % (stat, p))
if p > 0.05:
    print("Fail to reject null hypothesis")
else:
    print("Reject null hypothesis")

# %% [markdown]
# ## RQ3 (20 points)
#
# > Design and perform a statistical test involving neighbourhood and list price.
# > In a markdown cell, describe the null hypothesis statement of your test and
# > why you choose a specific statistical test method.
#
# Null hypothesis: Neighbourhood has no effect on listing price.
# Alternative hypothesis: Neighbourhood has an effect on listing price.
#
# Statistical test method: Spearman's Rank Correlation.
# Reasoning: I want to test the relationship between a listing's neighbourhood and its price. Since there's no independent variable, and I can't assume that each room neighbourhood has normally distributed prices, I decided to go with a non-parametric test.

# %%
neighbourhood_price_df = raw_listings_df[["neighbourhood", "price"]][filter]
descending_median_order = (
    neighbourhood_price_df.groupby(["neighbourhood"])["price"]
    .median()
    .sort_values(ascending=False)
    .index
)

sns.violinplot(
    x="price",
    y="neighbourhood",
    data=neighbourhood_price_df,
    order=descending_median_order,
)

# %%
stat, p = spearmanr(
    neighbourhood_price_df["neighbourhood"], neighbourhood_price_df["price"]
)

print("stat=%.3f, p=%.3f" % (stat, p))
if p > 0.05:
    print("Fail to reject null hypothesis")
else:
    print("Reject null hypothesis")

# %% [markdown]
# ## RQ4 (30 points)
#
# > Design and perform at least two correlation analyses
# > among attributes provided in the dataset. Describe why you believe it
# > would be interesting to analyze the correlation between selected attributes.
# > Write code and describe your main findings from the correlation analysis.

# %%
filtered_listings_df = raw_listings_df[filter]

# %%
sns.pairplot(data=filtered_listings_df)

# %% [markdown]
# ### Correlation between review count and calculated host listing count
#
# Without access to review scores, seeing a correlation between a hosts's experience and the number of reviews their listings have could reveal some insight on people's reasons for leaving reviews based on their quality of listings from experienced hosts.

# %%
first_attribute = "calculated_host_listings_count"
second_attrbute = "number_of_reviews"
print(
    filtered_listings_df[first_attribute].corr(
        filtered_listings_df[second_attrbute]
    )
)
sns.regplot(data=filtered_listings_df, x=first_attribute, y=second_attrbute)

# %% [markdown]
# It's interesting seeing a slightly negative correlation between a listing's review count and host's listings count. Hosts with a higher listing count would imply that they have more experience being hosts, and thus have better quality listings. Without access to review score data, seeing that their listings tend to have lower review counts could imply that the listings are higher quality, and so not as many people review the listing to complain.

# %% [markdown]
# ### Correlation between review count and price
#
# Does having high review counts make the listing seem popular, thus implying a sense of demand that warrants a higher price?

# %%
first_attribute = "price"
second_attrbute = "number_of_reviews"
print(
    filtered_listings_df[first_attribute].corr(
        filtered_listings_df[second_attrbute]
    )
)
sns.regplot(data=filtered_listings_df, x=first_attribute, y=second_attrbute)

# %% [markdown]
# The small positive correlation between a listing's review count and its price suggests that there are some hosts that set their prices higher based on the number of reviews their listings have. Although, it may also be a consequence of having more people live in the listings directly. In either case, one could say that host play a small game of economics (demand/price) based on their review count.
