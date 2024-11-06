# %%
import seaborn as sns
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("restaurant-1-orders.csv")
# %%
data.head()

# %%
data.isna().mean()
# %%
Min_quantity = data['Quantity'].min()
Max_Quantity = data['Quantity'].max()
print(Min_quantity)
print(Max_Quantity)

# %%

for col in data:
    if col == 'Item Name':
        unique = (data[col].unique())
        print(len(unique))
print(unique)

# %%
Item_unique = pd.DataFrame(unique)
Item_unique.columns = ['Name']
Item_unique.head()

# %%
Item_sorted = Item_unique.sort_values('Name', ignore_index=True)
Item_sorted.head()

# %%
#data.sort_values('Item Name', 'Order Date', ignore_index=True)
# %%
correlation_matrix = data.corr()
print(round(correlation_matrix, 2))


sns.heatmap(correlation_matrix)

# %%


# %%
new_df = data[['Item Name']].copy()


new_df.head()
# %%
Item = data.groupby('Item Name').count()

# %%
Item.head()

# %%
Item
# %%
Only_Item = Item.drop(
    ['Order Date', 'Quantity', 'Product Price', 'Total products'], axis=1)

# %%

Only_Item.rename(columns={'Order Number': 'Item count'}, inplace=True)

# %%
Only_Item
# %%

data
# %%


# %%
f_column = data["Product Price"]
data1 = pd.concat([Only_Item, f_column], axis=1)
data1
# %%
extracted_col = data["Product Price"]

Final = Only_Item.join(extracted_col)

Final
# %%
extracted_col.head()
# %%
data["date"] = pd.to_datetime(data['Order Date']).dt.date
data
# %%
# %%

# joining the product price column to the above created data frame

count_1 = count.merge(unique_data, on="Item Name", how="left")
count_1 = count_1[["Item Name", "Count", "Product Price"]]
count_1 = count_1.sort_values(by="Count", ascending=False)
count_1

# %%
sns.scatterplot(x="Count", y="Product Price", data=count_1)
plt.xlim(0, 1000)

# %%
corr, _ = pearsonr(count_1["Product Price"], count_1["Count"])
print('Pearsons correlation: %.3f' % corr)

# Greater the product price lower are the number of orders placed
