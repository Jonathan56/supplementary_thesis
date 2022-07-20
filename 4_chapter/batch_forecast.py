from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
import pickle

df = pd.read_pickle("fr_quoilin_data_valence.pickle")
nb_houses = 20
house_ids=[
    "2000989","2001197","2000914","2001123","2000964",
    "2001189","2001111","2001179","2000909","2000918",
    "2000994","2001107","2000913","2001139","2000960",
    "2001149","2001165","2000954","2001114","2000926",
]
df = df[house_ids + ["pv_1kw"]]
print(f"Average consumption is {round((df[house_ids].sum() * 15 / 60).mean() / 1000, 2)} MWh")
sub_house_ids = [house_ids[i] for i in [0, 4, 19, 13, 14]]
print(f"Average consumption is {round((df[sub_house_ids].sum() * 15 / 60).mean() / 1000, 2)} MWh")

##############################
_individuals = df.copy()
df["community_kW"] = df[house_ids].sum(axis=1)
df.drop(house_ids, axis=1, inplace=True)

##############################
individuals = _individuals.copy()

# Fix variables
deltat = timedelta(minutes=15)

# Training = 31 days + 1 day for lagged values
training = timedelta(days=31)

# Where do we forecast?
start = datetime(2019, 5, 30, 6, 0, 0)
full_horizon = timedelta(days=7)

# When do we calibrate
start_calibrate = start - timedelta(days=2)
end_calibrate = start - deltat
start_training_to_calibrate = start_calibrate - training
end_training_to_calibrate = start_calibrate - deltat

# To truncate data
end = start + full_horizon + timedelta(days=7)  # keep a few days after anyway

pv_size = 3
individuals = individuals.loc[start_training_to_calibrate-timedelta(days=1):end, :].copy()
for col in individuals.columns:
    individuals[col] -= pv_size * individuals["pv_1kw"]
individuals.drop(columns=["pv_1kw"], inplace=True)

##############################
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm

class GAM():
    """Generalized Additive Model.
    """

    def __init__(self, output, regressors=None,
                 daily_seasonality="auto",
                 seasonality_prior_scale=10.0):

        self._output = output
        self._model = Prophet(
            growth='flat',
            yearly_seasonality=False,
            weekly_seasonality="auto",
            daily_seasonality=daily_seasonality,
            seasonality_mode="additive",
            interval_width=0.95,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=seasonality_prior_scale,
            uncertainty_samples=False,
        )
        if regressors is None:
            regressors = list()
        for reg in regressors:
            self._model.add_regressor(
                name=reg["name"],
                prior_scale=reg["prior_scale"])

    def fit(self, df):
        with suppress_stdout_stderr():
            self._model.fit(self._specific_formatting(df))

    def predict(self, df):
        forecast = self._model.predict(self._specific_formatting(df))
        forecast.set_index("ds", inplace=True, drop=True)
        forecast.drop(columns=forecast.columns.difference(["yhat"]), inplace=True)
        forecast.rename(columns={"yhat": self._output}, inplace=True)
        return forecast

    def _specific_formatting(self, df):
        df = df.copy()
        df["ds"] = df.index.tz_localize(None)
        df.rename(columns={self._output: "y"}, inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def get_gof(df, result, ref_col, pred_col):
    """
    gof = (1 - NRMSE) * 100
    """
    pred = result.loc[:, [pred_col]].copy()
    #if pred.index.tzinfo is None:
    #    pred.index = pred.index.tz_localize("UTC")
    pred.columns = ["prediction"]

    ref = df.loc[pred.index[0]:pred.index[-1], [ref_col]].copy()
    ref.columns = ["target"]

    nrmse = (np.linalg.norm(ref["target"].values - pred["prediction"].values, 2)
           / np.linalg.norm(ref["target"].values - ref["target"].mean(), 2))
    return 100.0 * (1.0 - np.clip(nrmse, a_min=0.0, a_max=1.0))

def lag_values(df, nb_lag, output_col):
    tmp = df.copy()
    for shift in nb_lag:
        tmp[f"t-{shift}"] = tmp[output_col].shift(shift)
    return tmp

def predict_n_periods_with_autoreg(df, start_training, end_training, horizon,
                                   deltat, end_complete_pred, freq, output_col,
                                   regressors, nb_lag, seasonality_prior_scale=1.0,
                                   disable_progress_bar=False, daily_seasonality="auto"):
    """
    Train a GAM and predict for horizon T
    Shift prediction start and predict over T again.

    results : [pd.DataFrame] One frame per prediction.
    """

    inputs = [reg["name"] for reg in regressors]
    results = []
    model = GAM(output_col, regressors, daily_seasonality=daily_seasonality,
                seasonality_prior_scale=seasonality_prior_scale)

    tmp = lag_values(df.loc[start_training-timedelta(days=1):end_training], nb_lag, output_col)
    model.fit(tmp.loc[start_training:end_training])

    # Number of forecast where we have access to actual data
    forecast_freq = pd.date_range(end_training + deltat, end_complete_pred, freq=freq)
    for start_prediction in tqdm(forecast_freq, desc="# Forecast: ", disable=disable_progress_bar):
        tmp_results = []
        end_prediction = start_prediction + horizon

        # Get lagged values and NaN to blank future info
        tmp = lag_values(df.loc[start_prediction-timedelta(days=1):end_prediction], nb_lag, output_col)
        tmp = tmp.loc[start_prediction:end_prediction]
        for n in nb_lag:
            tmp.loc[:, f"t-{n}"] = tmp[f"t-{n}"].iloc[0:n].tolist() + ([np.nan] * (len(tmp) - n))

        horizon_spam = pd.date_range(start_prediction, end_prediction, freq="15T")
        for step_i, step in enumerate(horizon_spam):
            # Fill up NaN of lagged values with previous results
            for n in nb_lag:
                if pd.isna(tmp.at[step, f"t-{n}"]):
                    tmp.at[step, f"t-{n}"] = tmp_results[step_i-n]

            res = model.predict(tmp.loc[step:step, inputs])
            tmp_results.append(res.at[step, output_col])

        results.append(pd.DataFrame(index=horizon_spam, data={output_col: tmp_results}))
    return results, model

def model_3(graph, start_training, end_training, horizon,
            deltat, end_complete_pred, freq, output_col, scenario):

    regressors = [{"name": "pv_1kw", "prior_scale": scenario["PRIOR_GHI"]}]

    for hour in range(0, 23):
        regressors.append({"name": f"h{hour}", "prior_scale": scenario["PRIOR_HOUR"]})

    nb_lag = list(range(1, scenario["NB_LAG"] + 1))
    for n in nb_lag:
        regressors.append({"name": f"t-{n}", "prior_scale": scenario["PRIOR_LAG"]})

    results, _ = predict_n_periods_with_autoreg(
        graph, start_training, end_training, horizon, deltat, end_complete_pred, freq, output_col,
        regressors,
        nb_lag=nb_lag,
        seasonality_prior_scale=scenario["PRIOR_SEASON"],
        daily_seasonality=scenario["DAILY_FOURIER"],
        disable_progress_bar=True)
    return results

def calibrate(df, individuals, start_training, end_training, horizon, deltat, end_complete_pred, freq, output_col):
    gofs = []
    reference = {
         "NB_LAG": 4,
         "PRIOR_GHI": 3.0,
         "PRIOR_LAG": 10.0,
         "PRIOR_HOUR": 10.0,
         "PRIOR_SEASON": 1.0,
         "DAILY_FOURIER": "auto"}

    scenarios = [reference]
    for i in range(1, 14 + 1):  #  range(1, 24 + 1)
        scenarios.append(dict(reference))
        scenarios[-1]["NB_LAG"] = i

    for i in [1, 5, 8, 10, 15, 20]:
        scenarios.append(dict(reference))
        scenarios[-1]["PRIOR_GHI"] = i

    for i in [1, 5, 15]:  #  [1, 3, 5, 8, 15, 20]
        scenarios.append(dict(reference))
        scenarios[-1]["PRIOR_LAG"] = i

    for i in [8, 15]:  #  [1, 3, 5, 8, 15, 20]
        scenarios.append(dict(reference))
        scenarios[-1]["PRIOR_HOUR"] = i

    for i in [3]:  #  [3, 5, 8, 10, 15, 20]
        scenarios.append(dict(reference))
        scenarios[-1]["PRIOR_SEASON"] = i

    #for i in ["auto", 5, 10, 15, 20, 30]:  # no test
    #    scenarios.append(dict(reference))
    #    scenarios[-1]["DAILY_FOURIER"] = i

    graph = individuals[[output_col]].copy()
    graph["pv_1kw"] = df.loc[graph.index[0]:graph.index[-1], "pv_1kw"]

    graph["_datetime"] = graph.index
    for hour in range(0, 23):
        graph[f"h{hour}"] = graph._datetime.apply(lambda x: 1.0 if x.hour == hour else 0)
    graph.drop(columns="_datetime", inplace=True)

    for scenario in tqdm(scenarios, desc="Calibration :"):
        results = model_3(graph, start_training, end_training, horizon,
                    deltat, end_complete_pred, freq, output_col, scenario)

        results = pd.concat(results, axis=0)
        gofs.append(get_gof(individuals, results, output_col, output_col))
        #tmp_gof = []
        #for result in results:
        #    tmp_gof.append(get_gof(individuals, result, output_col, output_col))
        #gofs.append(np.mean(tmp_gof))

    results = pd.DataFrame(data=scenarios)
    results["gof"] = gofs
    return results

##############################
print("")
print(f"start_training_to_calibrate={start_training_to_calibrate}")
print(f"end_training_to_calibrate={end_training_to_calibrate}")
print(f"start_calibrate={start_calibrate}")
print(f"end_calibrate={end_calibrate}")
print("")

horizon = timedelta(hours=2, minutes=45)
freq = f"180T"

##############################
full_scores = []
best_scores = []
for house_id in sub_house_ids:
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        results = calibrate(df, individuals,
                            start_training_to_calibrate,
                            end_training_to_calibrate,
                            horizon,
                            deltat,
                            end_calibrate,
                            freq, house_id)

    score = results.copy()
    full_scores.append(score)
    best_scores.append(score.nlargest(columns="gof", n=1).to_dict('records')[0])
    print(best_scores[-1])
print("")
print(pd.DataFrame(index=sub_house_ids, data=best_scores))
best_parameters = pd.DataFrame(index=sub_house_ids, data=best_scores).T.to_dict()


##############################
start_training = start - training
end_training = start - deltat

horizon = timedelta(days=2) - deltat
end_complete_pred = start + timedelta(days=7)

freqs = ["2D", "1D", "12H", "6H", "3H", "1H", "15T"]
freq_deltas = [timedelta(days=2) - deltat,
               timedelta(days=1) - deltat,
               timedelta(hours=12) - deltat,
               timedelta(hours=6) - deltat,
               timedelta(hours=3) - deltat,
               timedelta(hours=1) - deltat,
               timedelta(minutes=15) - deltat
              ]

house_forecast = {}
for freq in freqs:
    print(f"Freq = {freq}")
    house_forecast[freq] = {}

    for house_id in tqdm(sub_house_ids, desc="House #"):

        graph = individuals[[house_id]].copy()
        graph["pv_1kw"] = df.loc[graph.index[0]:graph.index[-1], "pv_1kw"]

        graph["_datetime"] = graph.index
        for hour in range(0, 24):
            graph[f"h{hour}"] = graph._datetime.apply(lambda x: 1.0 if x.hour == hour else 0)
        graph.drop(columns="_datetime", inplace=True)

        results = model_3(graph, start_training, end_training, horizon,
                    deltat, end_complete_pred, freq, house_id, best_parameters[house_id])

        house_forecast[freq][house_id] = [res.copy() for res in results]
    with open(f'house_forecast_{freq}.pickle', 'wb') as handle:
        pickle.dump(house_forecast[freq], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("")

# Reformat as a dict of freq with a list of pd.DataFrame with all houses.
house_forecast_list = {}
for freq in freqs:
    house_forecast_list[freq] = []

    for i in range(0, len(house_forecast[freq][sub_house_ids[0]])):
        tmp = pd.concat([house_forecast[freq][house_id][i] for house_id in sub_house_ids], axis=1)
        house_forecast_list[freq].append(tmp)

# Store data (serialize)
with open('house_forecast_list.pickle', 'wb') as handle:
    pickle.dump(house_forecast_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
