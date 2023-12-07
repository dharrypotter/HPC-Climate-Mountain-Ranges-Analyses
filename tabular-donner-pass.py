import pandas as pd
import numpy as np
import seaborn as sns
from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt

# also need to pip install imodels


def load_csv_data(filename: str):
    df = pd.read_csv(filename)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.replace("--", np.nan)
    df = df.replace("T", np.nan)
    df = df.replace(".", np.nan)
    df["Air Temp Max (C)"] = df["Air Temp Max (C)"].astype(float)
    df["24-hour Total Precip (mm)"] = df["24-hour Total Precip (mm)"].astype(float)
    df["% of Precip as Snow"] = df["% of Precip as Snow"].astype(float)
    df["% of Precip as Rain"] = df["% of Precip as Rain"].astype(float)
    df["New Snow (cm)"] = df["New Snow (cm)"].astype(float)
    df["Snowpack depth (cm)"] = df["Snowpack depth (cm)"].astype(float)

    # clean and prepare your csv data here
    # handle missing values if necessary
    df = df.bfill()
    df = df.ffill()
    df = df.groupby(df["Date"].dt.to_period("D")).first()

    df.drop(
        columns=[
            "Snow Water Equivalent (cm)",
            "Season Total Snow (cm)",
            "Remarks",
            "Season Total Precip (mm)",
        ],
        inplace=True,
    )

    return df


# create a regression model
def tabular(target: str, df: pd.DataFrame, path=None):
    train_data = TabularDataset(df)

    if path is not None:
        try:
            predictor = TabularPredictor.load(path)
        except Exception:
            print(f"Saved model path {path} not found. Training new model")
            predictor = TabularPredictor(path=path, label=target)
            predictor.fit(
                train_data,
                presets="interpretable",
                time_limit=600,
            )
    else:
        print("Training new model")
        predictor = TabularPredictor(path=path, label=target)
        predictor.fit(
            train_data,
            presets="interpretable",
            time_limit=600,
        )

    # TODO: reserve some of the data for validation
    print(predictor.evaluate(train_data, silent=True))

    # the trained model can be used to predict future values


# create plots of the data
def plots(df):
    # correlations
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
    plt.tight_layout()

    # scatter plots
    sns.pairplot(df)

    plt.show()


data = load_csv_data("donner_pass_data.csv")
tabular("Air Temp Max (C)", data, path="donner-pass-tabular")
plots(data)
