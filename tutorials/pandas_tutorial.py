# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %% [markdown]
# # Loading data into Pandas

# %%
# Importing required packages.

import re
from os.path import dirname, join, realpath

import pandas as pd


# %%
def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    script_dir = dirname(realpath("__file__"))
else:
    script_dir = dirname(realpath(__file__))

# %%
# Reading the data into a Pandas DataFrame.
df = pd.read_csv(join(script_dir, "data/pokemon_data.csv"))

# %% [markdown]
# ## Reading Data in Pandas

# %%
# Read headers
# print(df.columns)

# Read each column
# print(df[["Name","Type 1", "HP"]][0:5])

# Read each row (iloc = integer location)
# print(df.iloc[0:4])

# Read rows iteratively.
# for index, row in df.iterrows():
#     print(index, row["Name"])

# Read rows based on column value.
# df.loc[df['Type 1'] == "Fire"]

# Read a spefic location (R, C)
# print(df.iloc[2, 1])

# %% [markdown]
# ## Sorting/Describing Data

# %%
# Generate descriptive statistics
# df.describe()

# Sorting
# df.sort_values(['Type 1', "HP"], ascending=[1,0])

# %% [markdown]
# ## Making changes to the data

# %%
# Creating new columns that are totals of stats others.
# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']

# df = df.drop(columns=["Total"])

df["Total"] = df.iloc[:, 4:10].sum(axis=1)

cols = list(df.columns.values)

df = df[cols[0:4] + [cols[-1]] + cols[4:12]]

df.head(5)

# %% [markdown]
# ## Saving our Data (Exporting into desired format)

# %%
df.to_csv(join(script_dir, "data/modified.csv"), index=False)
df.to_excel(join(script_dir, "data/modified.xlsx"), index=False)
df.to_csv(join(script_dir, "data/modified.txt"), index=False, sep="\t")

# %% [markdown]
# ## Filtering Data

# %%
# new_df = df.loc[(df["Type 1"] == "Grass") & (df["Type 2"] == "Poison") & (df["HP"] > 70)]

# new_df = df.loc[~df["Name"].str.contains("Mega")]

# Regex
new_df = df.loc[df["Name"].str.contains("^pi[a-z]*", flags=re.I, regex=True)]

new_df.reset_index(drop=True, inplace=True)

new_df

# %% [markdown]
# ## Conditional changes

# %%
df.loc[df["Type 1"] == "Fire", "Legendary"] = "True"

df

# %%
df.loc[df["Total"] > 500, ["Generation", "Legendary"]] = ["Test 1", "Test 2"]

df

# %% [markdown]
# ## Aggregate statistics (groupby)

# %%
df = pd.read_csv(join(script_dir, "data/modified.csv"))

# %%
# df.groupby(["Type 1"]).mean().sort_values("HP", ascending=False)

# df.groupby(["Type 1"]).sum()

df["count"] = 1
df.groupby(["Type 1", "Type 2"]).count()["count"]

# %% [markdown]
# ## Working with large amounts of data

# %%
new_df = pd.DataFrame(columns=df.columns)

for df in pd.read_csv(join(script_dir, "data/modified.csv"), chunksize=5):
    # print("CHUNK DF")
    # print(df)

    results = df.groupby(["Type 1"]).count()
    new_df = pd.concat([new_df, results])

new_df
