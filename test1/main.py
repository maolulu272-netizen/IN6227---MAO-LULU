import pandas as pd

df = pd.read_csv("test.csv")

'''print("数据规模：", df.shape)
print("列名：", df.columns.tolist())
print("前 5 行数据：")
print(df.head())

print("每列的数据类型和非空数量：")
df.info()

print("\n每列缺失值数量：")
print(df.isna().sum())

print("label 各类别的数量：")
print(df["label"].value_counts(dropna=False))

print("\nlabel 各类别的比例：")
print(df["label"].value_counts(normalize=True, dropna=False).round(4))
'''

print(df.describe().round(2))