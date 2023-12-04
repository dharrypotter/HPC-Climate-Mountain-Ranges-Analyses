import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt

# also need to pip install imodels


def load_csv_data(filename: str):
    df = pd.read_csv(filename)

    df["date"] = pd.to_datetime(df["date"])

    # clean and prepare your csv data here
    # handle missing values if necessary
    df = df.bfill()
    df = df.ffill()
    df = df.groupby(df["date"].dt.to_period("D")).first()

    df.drop(
        columns=[
            "barograph_reading",
            "barograph_correction",
            "snow_depth_trace",
            "snow_fall_trace",
            "precipitation_trace",
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


data = load_csv_data("White_Mountains/1939_to_2022_b16_synoptic_data.csv")
tabular("precipitation", data, path="white-mountains-1939-2022-precip")
plots(data)
