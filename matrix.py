import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from os import path
from datetime import datetime, date, timedelta

from mock_shiny_inputs import Input
from calc import *

input = Input()

df_fondsen = pd.read_csv("fondsen.csv")
df_fondsen = df_fondsen.dropna()
df_fondsen = df_fondsen.set_index("id")
df_fondsen

def df_mnd_stortingen(df):
    storting_datums = df[df["Datum"].dt.day >= input.in_dag_storting()].groupby("Maand")["Datum"].min()
    storting_datums

    df_storting_datums = df[df["Datum"].isin(storting_datums)].copy()
    df_storting_datums["Aandelen"] = input.in_bedrag_storting() / df_storting_datums["Koers"]

    return df_storting_datums


def df_mnd_verkoop_datums(df):
    verkoop_datums = df[df["Datum"].dt.day >= input.in_dag_verkoop()].groupby("Maand")["Datum"].min()

    df_verkoop_dt = df[df["Datum"].isin(verkoop_datums)].copy()

    return df_verkoop_dt


for f_id in df_fondsen.index:
    file_name = df_fondsen.loc[f_id, "file_name"]
    directory = df_fondsen.loc[f_id, "directory"]
    name = df_fondsen.loc[f_id, "name"]

    save_path = path.join("Pre", "{}_{}.csv".format(f_id, input.in_instapkost_pct()))

    file_path = path.join("Data", directory, file_name)

    if path.exists(save_path):
        continue

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "Datum", "Price": "Koers"})
    df = df[["Datum", "Koers"]]
    if df["Koers"].dtype == "object":
        df["Koers"] = df["Koers"].str.replace(",", "").astype(float)
    df["Maand"] = df["Datum"].apply(lambda x: x.replace(day=1))

    print(df.head())
    df_stortingen = df_mnd_stortingen(df)

    df_verkoop = df_mnd_verkoop_datums(df)
    df_stortingen_verkoop = pd.merge(df_stortingen["Datum"], df_verkoop["Datum"], how="cross",
                                     suffixes=("_Eerste_Storting", "_Verkoop"))
    df_stortingen_verkoop = df_stortingen_verkoop[
        df_stortingen_verkoop["Datum_Verkoop"] > df_stortingen_verkoop["Datum_Eerste_Storting"]]

    df_stortingen_verkoop[["Investering Totaal","Verkoop Totaal","Winst Pct", "Eff Interest"]] = df_stortingen_verkoop.apply(lambda x: ji(df,
                     df_stortingen,
                     x["Datum_Eerste_Storting"],
                     x["Datum_Verkoop"],
                     input.in_instapkost_pct(),
                     input.in_verkoop_sper_periode(),
                     input.in_min_stortingen_voor_verkoop()), axis=1, result_type="expand")

    print(df_stortingen_verkoop)
    print("Saving to {}".format(save_path))
    df_stortingen_verkoop.to_csv(save_path)

