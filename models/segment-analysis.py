from tqdm import tqdm
import pandas as pd
import os

base_dir = os.path.join("datasets", "IMDB")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

#%%

def load_reviews(directory):
    reviews = []
    labels = []
    for label_type in ["pos", "neg"]:
        label_dir = os.path.join(directory, label_type)
        file_names = os.listdir(label_dir)
        
        for file_name in tqdm(file_names, desc=f"Loading {label_type} reviews"):
            if file_name.endswith(".txt"):
                with open(os.path.join(label_dir, file_name), encoding="utf-8") as file:
                    reviews.append(file.read())
                    labels.append(1 if label_type == "pos" else 0)
    return reviews, labels

train_reviews, train_labels = load_reviews(train_dir)
test_reviews, test_labels = load_reviews(test_dir)

#%%

train_df = pd.DataFrame({'review': train_reviews, 'label': train_labels})
test_df = pd.DataFrame({'review': test_reviews, 'label': test_labels})

#%%

dataFrame = pd.concat([train_df, test_df])
print(dataFrame.shape)
print(dataFrame.head(1))
  
#%%