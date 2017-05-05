import pandas as pd

data = pd.read_csv("./data/winequality-white.csv",
                   delimiter=";")

y = data["quality"]
X = data.drop("quality", axis=1)

# Below, call OLS on y and X. You may access the data itself using y.values, and
# X.values (those return a vector/matrix with just the data and not the column
# names.

from models import OLS

model = OLS(x, y)
