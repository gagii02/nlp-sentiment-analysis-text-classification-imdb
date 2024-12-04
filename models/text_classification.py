import os
from sklearn.model_selection import train_test_split
import pandas as pd
import utils

folder_path = os.path.join("..", "datasets", "20_newsgroups")

#%%

X, y, target_names = utils.load_data_from_folder(folder_path)

#%%

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

x_train_df = pd.DataFrame(x_train, columns=["Text"])

y_train_df = pd.DataFrame(y_train, columns=["Label"])

print(x_train_df.head(1))
print(y_train_df.head(5))

print(folder_path)

#%%
