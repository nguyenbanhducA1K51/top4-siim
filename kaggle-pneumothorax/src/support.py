import pandas as pd
from sklearn.model_selection import KFold

# csv1="/root/data/siim_png_convert/train.csv"
# csv2="/root/data/siim_png_convert/val.csv"
# csv3="/root/data/siim_png_convert/test.csv"
# df1=pd.read_csv(csv1)
# df2=pd.read_csv(csv2)
# df3=pd.read_csv(csv3)
# df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
# df = pd.concat([df, df3], axis=0).reset_index(drop=True)

# # Specify the number of folds (k)
# k = 5


# kf = KFold(n_splits=k, shuffle=True, random_state=42)

# # # Iterate through the folds and assign fold numbers to the DataFrame
# for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
#     df.loc[val_idx, 'Fold'] = fold
# df['Fold'] = df['Fold'].astype(int)
# df.to_csv("/root/data/siim_png_convert/k_fold.csv")
# Display the DataFrame with fold numbers
# print(df.head())

def foo():
    for i in range (5):
        yield i
for x in foo():
    p

