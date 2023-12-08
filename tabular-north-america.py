import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt

# also need to pip install imodels


def load_csv_data(filename: str):
    df = pd.read_csv(filename)

    df["Time"] = pd.to_datetime(df["Time"])

    # clean and prepare your csv data here
    # handle missing values if necessary
    df = df.bfill()
    df = df.ffill()
    df = df.groupby(df["Time"].dt.to_period("D")).first()

    df.drop(
        columns=[
            "Latitude",
            "Longitude",
            "LatitudeRegion",
            "LongitudeRegion",
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


data = load_csv_data("addy_total_average_north_america.csv")
# tabular("temperature_max", data, path="total-average-north-america-tabular")
plots(data)
