# %%
import datetime
from IPython.display import display
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# %%


data = pd.read_csv("restaurant-1-orders.csv")

# %%
data.head()
# %%
item_freq = data.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.tail(20)
top_20.plot(kind="barh", figsize=(16, 8))
plt.title('Top 20 sold items')
# %%
top_20.plot(kind="pie", figsize=(16, 8), subplots=True, legend=None)
plt.title('Top 20 sold items')
# %%


sns.relplot(data=top_20, kind='line')

# %%
top_20.plot(kind="hist", figsize=(16, 8), subplots=True, legend=None)
plt.title('Top 20 sold items')

# %%


# %%
item_freq = data.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.head(20)
top_20.plot(kind="barh", figsize=(16, 8))
plt.title('Least 20 sold items')

# %%
sns.lineplot(data=top_20, y="Item Name", x='Quantity')

# %%
top_20.plot(kind="pie", figsize=(16, 8), subplots=True, legend=None)
plt.title('Top 20 sold items')

# %%
data['hour'] = data['Order Date'].dt.hour
data.sample(5)
# %%
data['date'] = data['Order Date'].dt.strftime(
    '%y/%m/%d')  # Add column with date
data.sample(5)

#
# %%
# %%


def avg_hour(hour):
    by_hour = data[data['hour'] == hour]
    avg = len(by_hour['Order Number'].unique()) / len(data['date'].unique())
    return avg


hours = pd.DataFrame(sorted(data['hour'].unique()))

hours.rename(columns={0: 'hour'}, inplace=True)
hours['Average orders'] = hours['hour'].apply(avg_hour)

hours.set_index('hour', inplace=True)
hours.head()

# %%
hours.plot.bar(figsize=(11, 6), rot=0)
plt.xlabel('Hour')
plt.title('Average number of orders by hour of the day')

# %%
sns.scatterplot(data=data, x="hour", y="Quantity", s=5)
sns.rugplot(data=data, x="hour", y="Quantity", lw=1, alpha=.005)

# %%
sns.pointplot(data=data, x="hour", y="Quantity")

# %%
sns.boxenplot(data=data, x="hour", y="Quantity")

# %%
sns.swarmplot(data=data, x="hour", y="Quantity")

# %%
