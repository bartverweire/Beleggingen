import numpy as np
import pandas as pd

from datetime import timedelta

def eff_interest(aankoop_koers, instapkost_pct, verkoop_koers, dagen_actief):
    if dagen_actief == 0:
        return 0

    return 100*(((1 - instapkost_pct/100) * verkoop_koers / aankoop_koers) ** (365 / dagen_actief) - 1)

def bereken_winst_pct_vaste_interest(df, jaar_interest_pct):
    df_tmp = df.copy()

    dag_interest_pct = 100 * ((1 + jaar_interest_pct / 100) ** (1 / 365) - 1)

    # relatieve investering : aantal stortingen (onafhankelijk van bedrag van de storting)
    investering_totaal = df.shape[0]

    # relative interest per storting
    df_tmp["Ratio vaste interest"] = (1 + dag_interest_pct / 100) ** df_tmp["Dagen Actief"]
    # totaal relatief bedrag na vaste interest
    vaste_interest_totaal = df_tmp["Ratio vaste interest"].sum()

    # winst percentage
    winst_pct_vaste_interest = 100 * (vaste_interest_totaal - investering_totaal) / investering_totaal

    return winst_pct_vaste_interest


def bereken_equiv_jaar_interest(df, n, eps=1e-5):
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan

    investering_totaal = df.shape[0]

    # bereken target winst_pct
    verkoop_totaal = df["Verkoop Ratio"].sum()
    # print("Investering: {}, Verkoop totaal: {}".format(investering_totaal, verkoop_totaal))

    winst_pct = 100 * (verkoop_totaal - investering_totaal) / investering_totaal

    # print("Winst pct: {}".format(winst_pct))

    int_a = -1
    int_b = 1

    fa = 1
    fb = 1

    # print("Begin en eind interest")
    lim_index = 1
    while fa * fb >= 0:
        int_a = 2 * int_a
        int_b = 2 * int_b

        if np.max(np.abs([int_a, int_b])) > 50:
            break

        try:
            fa = bereken_winst_pct_vaste_interest(df, int_a) - winst_pct
            fb = bereken_winst_pct_vaste_interest(df, int_b) - winst_pct
        except:
            print("Error bereken_winst_pct_vaste_interest voor int_a of int_b: {}, {}".format(int_a, int_b))
            print(df.info())
            raise Exception("Error bereken_winst_pct_vaste_interest voor int_a of int_b: {}, {}".format(int_a, int_b))

        # print("Loop {} - int_a: {}, int_b: {}, fa: {} ({}), fb: {} ({})".format(lim_index, int_a, int_b, fa, fb,
        #                                                                         fa - winst_pct, fb - winst_pct))

    # print("Start bissectie")
    err = eps + 1

    X = np.zeros(n)
    fX = np.zeros(n)
    i = 0

    while err > eps and i < n:
        int_c = (int_a + int_b) / 2
        try:
            fc = bereken_winst_pct_vaste_interest(df, int_c) - winst_pct
        except:
            print("Error bereken_winst_pct_vaste_interest voor int_c: {}".format(int_c))
            print(df.info())
            raise Exception("Error bereken_winst_pct_vaste_interest voor int_c: {}".format(int_c))

        err = np.abs(fc)

        # shortcut if interest diverges
        # if np.abs(int_c) > 20:
        #    print("Error ! Interest too big")
        #    err = 0
        #    int_x = int_c
        #    ferr_x = fc

        if fc == 0:
            int_x = int_c
            ferr = 0
            err = 0

        elif fa * fc > 0:
            # a en c zelfde teken
            int_a = int_c

            i_min = np.argmin(np.abs([fc, fb]))
            ferr_x = [fc, fb][i_min]
            int_x = [int_c, int_b][i_min]

        else:
            # b en c zelfde teken
            int_b = int_c

            i_min = np.argmin(np.abs([fa, fc]))
            ferr_x = [fa, fc][i_min]
            int_x = [int_a, int_c][i_min]

        i += 1

    # print("Resultaat: jaar interest {}, error {}".format(int_x, ferr_x))

    return investering_totaal, verkoop_totaal, winst_pct, int_x

def stortingen_voor_verkoop(df_koersen, df_stortingen, verkoop_dt, instapkost_pct, verkoop_sper_periode):
    # print("Stortingen voor verkoop - df_stortingen")
    # print(df_stortingen.head())
    # print("verkoop_dt : {}".format(verkoop_dt))
    df_geldige_stortingen = df_stortingen[df_stortingen["Datum"] < verkoop_dt - timedelta(days=verkoop_sper_periode)].copy()

    # print(df_verkoop.head())
    # print("Verkoop koers: ")
    verkoop_koers = koers_voor_datum(df_koersen, verkoop_dt)
    # print(verkoop_koers)

    df_geldige_stortingen["Inv Koers"] = 1 / df_geldige_stortingen["Koers"]
    df_geldige_stortingen["Verkoop Ratio"] = (1 - instapkost_pct / 100) * verkoop_koers / df_geldige_stortingen["Koers"]
    df_geldige_stortingen["Dagen Actief"] = (verkoop_dt- df_geldige_stortingen["Datum"]).dt.days

    return df_geldige_stortingen

def koers_voor_datum(df, datum):
    # print("datum: {}".format(datum))
    # print("datum type : {}".format(type(datum)))

    koers_df = df[df["Datum"] == datum]["Koers"]
    if not koers_df.empty:
        koers = koers_df.iloc[0]
    else:
        print("Geen koers voor datum: {}".format(datum))
        koers = 0

    # print("Koers voor datum: {}".format(koers))
    # print("Koers voor datum type : {}".format(type(koers)))
    return koers

def equiv_jaar_interest_voor_verkoop(df_koersen, df_stortingen, verkoop_dt, instapkost_pct, verkoop_sper_periode):
    df_geldige_stortingen = stortingen_voor_verkoop(df_koersen, df_stortingen, verkoop_dt, instapkost_pct, verkoop_sper_periode)

    (a, b, c, d) = bereken_equiv_jaar_interest(df_geldige_stortingen, 50, 1e-3)
    #print("Res bereken_equiv_jaar_interest: ({},{},{},{})".format(a, b, c, d))

    return (a, b, c, d)


def ji(df_koersen, df_stortingen, dt_eerste_storting, dt_verkoop, instapkost_pct, verkoop_sper_periode, min_stortingen_voor_verkoop):
    # print("datum eerste storting: {}".format(dt_eerste_storting))
    # print("datum verkoop: {}".format(dt_verkoop))
    # df_stortingen = df_mnd_stortingen()
    # print("Stortingen")
    # print(df_stortingen)
    df_geldige_stortingen = df_stortingen[
        (df_stortingen["Datum"] >= dt_eerste_storting) & (df_stortingen["Datum"] < dt_verkoop)]
    # print("Geldige stortingen")
    # print(df_geldige_stortingen)
    if len(df_geldige_stortingen) >= min_stortingen_voor_verkoop:
        investering_totaal, verkoop_totaal, winst_pct, eff_interest = equiv_jaar_interest_voor_verkoop(df_koersen, df_geldige_stortingen, dt_verkoop,
                                                        instapkost_pct,
                                                        verkoop_sper_periode)

        #print("res equiv_jaar_interest_voor_verkoop: {},{},{},{}".format(investering_totaal, verkoop_totaal, winst_pct, eff_interest))
        # print("eff interest voor verkoop_dt {} en eerste_storting {}: {}".format(dt_verkoop, dt_eerste_storting, eff_interest))

        return pd.Series([investering_totaal, verkoop_totaal, winst_pct, eff_interest])
    else:
        # print("Niet voldoende geldige stortingen")
        return pd.Series([np.nan, np.nan, np.nan, np.nan])