import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt


def load_csv_data(filename: str):
    df = pd.read_csv(filename)

    # clean and prepare your csv data here
    # need a datetime column with values in datetime format
    df = df[["date", "temp_max"]]
    df["date"] = pd.to_datetime(df["date"])

    # need this column for TimeSeriesPredictor
    df["item_id"] = 0

    # time series predictor won't work if there are missing values
    # fill them in using your preferred method

    # df.dropna(subset=[targets[0]], inplace=True)
    # df = df.interpolate(method="linear")
    df = df.bfill()
    df = df.ffill()
    df = df.groupby(df["date"].dt.to_period("D")).first()
    print(df.head())

    return df


def time_series_analysis(target: str, df: pd.DataFrame, path=None):
    """
    Performs time series analysis on the given DataFrame.

    :param target: The target variable to be predicted.
    :type target: str
    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param path: The path to load a saved model. If None, a new model will be trained.
    :type path: str, optional

    """
    # use TimeSeriesDataFrame.to_regular_index() to fill in gaps if necessary
    # see pandas documentation for frequency options
    # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    # then use TimeSeriesDataFrame.fill_missing_values() to fill in the resulting NaNs
    train_data = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="date"
    )
    # train_data = train_data.to_regular_index(freq="D")
    # train_data = train_data.fill_missing_values()

    if path is not None:
        try:
            predictor = TimeSeriesPredictor.load(path)
        except Exception:
            print(f"Saved model path {path} not found. Training new model")
            predictor = TimeSeriesPredictor(
                prediction_length=24,
                path=path,
                target=target,
                eval_metric="MASE",
                # ignore_time_index=True,
            )
            predictor.fit(
                train_data,
                presets="fast_training",
                time_limit=600,
            )
    else:
        print("Training new model")
        predictor = TimeSeriesPredictor(
            prediction_length=24,
            path=path,
            target=target,
            eval_metric="MASE",
            # ignore_time_index=True,
        )
        predictor.fit(
            train_data,
            presets="fast_training",
            time_limit=600,
        )

    predictions = predictor.predict(train_data)

    # TODO: reserve some of the data for validation,
    # put this data in the plot and use it for model leaderboard
    plt.figure(figsize=(20, 3))

    item_id = 0
    y_past = train_data.loc[item_id][target]
    y_pred = predictions.loc[item_id]

    plt.plot(y_past, label="Past time series values")
    plt.plot(y_pred["mean"], label="Mean forecast")

    plt.fill_between(
        y_pred.index,
        y_pred["0.1"],
        y_pred["0.9"],
        color="red",
        alpha=0.1,
        label="10%-90% confidence interval",
    )
    plt.xlabel("Time")
    plt.ylabel(target)
    plt.title(f"{target} forecast")
    plt.legend()
    plt.show()


time_series_analysis(
    "temp_max",
    load_csv_data("White_Mountains/1939_to_2022_b16_synoptic_data.csv"),
    path="white-mountains-synoptic-1939-2022",
)
