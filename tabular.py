import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt

# also need to pip install imodels


def load_csv_data(filename: str):
    df = pd.read_csv(filename)

    # clean and prepare your csv data here
    # handle missing values if necessary
    df.drop(
        columns=[
            "year",
            "weather.n",
            "snow.year",
            "month",
            "no.gdd",
            "mean.daylength",
            "mean.soiltemp.20in.C",
            "mean.soilmoisture.20in",
            "mean.flow.ft3s.cb",
            "delta.mean.temp",
        ],
        inplace=True,
    )
    df = df.interpolate(method="linear")

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
    sns.heatmap(df.corr(), cmap="coolwarm")

    # scatter plots
    sns.pairplot(df)

    plt.show()


data = load_csv_data("RMBL-weather-months-1974-2022.csv")
# data = data[["mean.temp.C.cb", "min.temp.C.cb", "max.temp.C.cb", "precip.mm.cb.bb"]]
# tabular("precip.mm.cb.bb", data, path="RMBL-tabular-precip")
plots(data)
