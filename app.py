from shiny import *
from shinywidgets import *
from datetime import datetime, date, timedelta
from os import path
from random import choices

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import qgrid

from calc import *

df_fondsen = pd.read_csv("fondsen.csv")
df_fondsen = df_fondsen.dropna()
df_fondsen = df_fondsen.set_index("id")

print(df_fondsen.head())
# df_koers_data_all = None
#
# for fnd in df_fondsen.index:
#     file_name = df_fondsen.loc[fnd, "file_name"]
#     directory = df_fondsen.loc[fnd, "directory"]
#     name = df_fondsen.loc[fnd, "name"]
#
#     file_path = path.join("Data", directory, file_name)
#
#     df = pd.read_csv(file_path, parse_dates=["Date"])
#     df = df.rename(columns={"Date": "Datum", "Price": "Koers"})
#     df = df[["Datum", "Koers"]]
#     df["Fonds"] = fnd
#     df["Fonds Naam"] = name
#
#     if df["Koers"].dtype == "object":
#         df["Koers"] = df["Koers"].str.replace(",", "").astype(float)
#     datum0 = df["Datum"].min()
#     koers0 = df[df["Datum"] == datum0]["Koers"]
#     print(f"datum0 {datum0} - koers0 {koers0}")
#     df["Relatieve Koers"] = df["Koers"].apply(lambda x: x/koers0)
#     df["Koers Delta"] = 100 * (df["Koers"] - df["Koers"].shift(1)) / df["Koers"].shift(1)
#     df["Koers Delta Min"] = df["Koers Delta"].rolling(7, min_periods=3, center=True).min()
#     df["Koers Delta Max"] = df["Koers Delta"].rolling(7, min_periods=3, center=True).max()
#
#     print(df.head())
#     df["Maand"] = df["Datum"].apply(lambda x: x.replace(day=1))
#
#     if df_koers_data_all is None:
#         df_koers_data_all = df
#     else:
#         df_koers_data_all = pd.concat([df_koers_data_all, df])
#
#
# df_koers_data_all.to_pickle("df_koers_data_all.pkl")
df_koers_data_all = pd.read_pickle("df_koers_data_all.pkl")
# print(df_koers_data_all.head())
# df_eff_jaar_interest = None
#
# for fonds in df_fondsen.index:
#     dff = pd.read_csv("Pre\\{}_3.csv".format(fonds), parse_dates=["Datum_Eerste_Storting", "Datum_Verkoop"])
#     dff["Fonds"] = fonds
#     dff["Fonds Naam"] = df_fondsen["name"].loc[fonds]
#
#     if df_eff_jaar_interest is None:
#         df_eff_jaar_interest = dff
#     else:
#         df_eff_jaar_interest = pd.concat([df_eff_jaar_interest, dff])
#
# df_eff_jaar_interest["Jaar Eerste Storting"] = df_eff_jaar_interest["Datum_Eerste_Storting"].dt.year
# df_eff_jaar_interest["Jaar Verkoop"] = df_eff_jaar_interest["Datum_Verkoop"].dt.year
# df_eff_jaar_interest.to_pickle("df_eff_jaar_interest.pkl")
df_eff_jaar_interest = pd.read_pickle("df_eff_jaar_interest.pkl")

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_numeric("in_bedrag_storting", "Bedrag Storting", min=1, max=10000, value=1000),
            ui.input_numeric("in_instapkost_pct", "Instapkost (%)", min=0, max=5, step=0.1, value=3),
            ui.input_numeric("in_verkoop_sper_periode", "Sper Periode", min=1, max=365, value=90),
            ui.input_numeric("in_min_stortingen_voor_verkoop", "Minimum # Stortingen", min=1, max=10, value=2),
            ui.input_numeric("in_dag_storting", "Dag Storting", min=1, max=25, step=1, value=10),
            ui.input_numeric("in_dag_verkoop", "Dag Verkoop", min=1, max=25, step=1, value=20),
            ui.input_select("in_fonds", "Fonds", choices=[], selected=[], multiple=False, selectize=True),
            width=2
        ),
        ui.panel_main(
            ui.navset_tab(
                ui.nav(
                    "Eenmalige Storting",
                    ui.h4("Effective Interest"),
                    output_widget("out_eff_interest_eenmalig"),
                ),
                ui.nav(
                    "Effectiviteitsmatrix",
                    ui.h4("Effective Interest"),
                    ui.tags.p("""
                    Onderstaande grafiek geeft de effectieve jaarinterest weer, voor maandelijkse stortingen startend vanaf 
                    Datum Eerste Storting (x-as), en verkocht op Datum Verkoop (y-as).
                    Enkel effectieve jaarinteresten tussen -15% en 15% worden getoond. Negatieve interesten in rood, positieve interesten in groen.
                    """),
                    ui.tags.p("Opgelet! Het kan enige tijd duren voor deze data berekend is"),
                    output_widget("out_eff_interest_matrix"),
                    output_widget("out_eff_interest_density")
                ),
                ui.nav(
                    "Vergelijking",
                    ui.h4("Vergelijking fondsen"),
                    ui.tags.p(
                        "Kies 2 fondsen. In de grafiek zullen de 2 fondsen vergeleken worden voor de situaties",
                        ui.tags.ul(
                            ui.tags.li("Maandelijkse stortingen vanaf jaar x, eerste beursdag na de 10e van de maand (stortingen kunnen beginnen in januari, februari, ..."),
                            ui.tags.li("Verkoop in jaar y, eerste beursdag na de 20e van de maand (verkoop kan gebeuren in januari, februari, ..."),
                            ui.tags.li("Voor elke combinatie (storting vanaf jaar x, verkoop in jaar y), is er een distributie van effectieve jaarinteresten"),
                            ui.tags.li("Voor elk fonds is een instapkost van 3% in rekenening gebracht (wat allicht niet correct is, maar de vergelijking gaat dus over de pure koerseffectiviteit)")
                        )
                    ),
                    ui.panel_well(
                        ui.input_select("in_fonds_vgl", "Fonds", choices=[], selected=[], multiple=True, selectize=True),
                    ),
                    output_widget("out_koers_vergelijking"),
                    output_widget("out_koers_delta_vergelijking"),
                    output_widget("out_eff_interest_vergelijking")
                )
            ),
            width=10
        )
    )
)

def server(input, output, session):

    verkoop_dt_doel = reactive.Value()

    @reactive.Effect
    def init_verkoop_dt():
        req(not data().empty)

        verkoop_dt_doel.set(data()["Datum"].max())

    @reactive.Effect
    def update_fonds_input():
        fondsen_dict = df_fondsen["name"].to_dict()
        ui.update_select("in_fonds", choices=fondsen_dict, selected=df_fondsen.index[0])
        ui.update_select("in_fonds_vgl", choices=fondsen_dict, selected=df_fondsen.index[0])


    @reactive.Calc
    def df_fonds_matrix():
        req(input.in_fonds())

        file_path = path.join("Pre", "{}_{}.csv".format(input.in_fonds(), input.in_instapkost_pct()))
        dff = pd.read_csv(file_path, parse_dates=["Datum_Eerste_Storting","Datum_Verkoop"])
        dff = dff.dropna()
        dff = dff[np.abs(dff["Eff Interest"]) <= 15]

        dff["Jaar Eerste Storting"] = dff["Datum_Eerste_Storting"].dt.year

        return dff

    @reactive.Calc
    def data():
        req(input.in_fonds())
        file_name = df_fondsen.loc[input.in_fonds(), "file_name"]
        directory = df_fondsen.loc[input.in_fonds(), "directory"]
        name = df_fondsen.loc[input.in_fonds(), "name"]

        file_path = path.join("Data", directory, file_name)

        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df.rename(columns={"Date": "Datum", "Price": "Koers"})
        df = df[["Datum", "Koers"]]
        if df["Koers"].dtype == "object":
            df["Koers"] = df["Koers"].str.replace(",", "").astype(float)
        df["Maand"] = df["Datum"].apply(lambda x: x.replace(day=1))

        return df

    @reactive.Calc
    def koers_data_sel():
        req(input.in_fonds_vgl())

        df = df_koers_data_all[df_koers_data_all["Fonds"].isin(input.in_fonds_vgl())]

        return df

    @reactive.Calc
    def verkoop_dt():
        req(not data().empty, verkoop_dt_doel())

        df = data()
        verkoop_dt = df[df["Datum"] >= verkoop_dt_doel()]["Datum"].min()

        print(f"Werkelijke verkoop datum: {verkoop_dt}")

        return verkoop_dt

    @reactive.Calc
    def verkoop_koers():
        req(verkoop_dt())

        df = data()

        print("verkoop_dt: {}, {}".format(verkoop_dt(), type(verkoop_dt())))
        df_verkoop = df[df["Datum"] == verkoop_dt()]

        verkoop_koers = df_verkoop["Koers"].iloc[0]
        print(f"Verkoop koers: {verkoop_koers}")

        return verkoop_koers


    @reactive.Calc
    def df_eff_interest():
        req(not data().empty, verkoop_dt())

        df = data().copy()
        df = df[df["Datum"] < verkoop_dt() - timedelta(days=input.in_verkoop_sper_periode())]

        df["Dagen Actief"] = (verkoop_dt() - df["Datum"]).dt.days
        df["Eff Interest"] = df.apply(
            lambda x: eff_interest(x["Koers"], input.in_instapkost_pct(), verkoop_koers(), x["Dagen Actief"]), axis=1
        )

        return df

    @reactive.Calc
    def df_mnd_stortingen():
        req(not data().empty,  input.in_dag_storting())

        df = data().copy()

        storting_datums = df[df["Datum"].dt.day >= input.in_dag_storting()].groupby("Maand")["Datum"].min()
        storting_datums

        df_storting_datums = df[df["Datum"].isin(storting_datums)].copy()
        df_storting_datums["Aandelen"] = input.in_bedrag_storting() / df_storting_datums["Koers"]

        return df_storting_datums

    @reactive.Calc
    def df_mnd_verkoop_datums():
        req(not data().empty,  input.in_dag_storting())

        df = data().copy()

        verkoop_datums = df[df["Datum"].dt.day >= input.in_dag_verkoop()].groupby("Maand")["Datum"].min()

        df_verkoop_dt = df[df["Datum"].isin(verkoop_datums)].copy()

        return df_verkoop_dt

    @reactive.Calc
    def df_stortingen_voor_verkoop():
        req(not df_mnd_stortingen().empty, verkoop_dt(), input.in_instapkost_pct())

        df_verkoop = stortingen_voor_verkoop(data(), df_mnd_stortingen(), verkoop_dt(), input.in_instapkost_pct(), input.in_verkoop_sper_periode())

        df_verkoop["Aandelen"] = input.in_bedrag_storting() / df_verkoop["Koers"]
        df_verkoop["Verkoop Waarde"] = input.in_bedrag_storting() * df_verkoop["Verkoop Ratio"]

        return df_verkoop

    @reactive.Calc
    def df_geldige_koersen():
        req(not data().empty, not df_mnd_stortingen().empty)

        df_koersen = data().copy()
        df_stortingen = df_mnd_stortingen().copy()
        min_stortingen = input.in_min_stortingen_voor_verkoop()

        min_verkoop_dt = df_stortingen["Datum"].sort_values().iloc[0:min_stortingen].max() + timedelta(days=input.in_verkoop_sper_periode())

        df_koersen = df_koersen[df_koersen["Datum"] > min_verkoop_dt]

        return df_koersen


    @reactive.Calc
    def df_koersen_met_equiv_jaar_interest():
        req(not df_geldige_koersen().empty, not df_mnd_stortingen().empty)

        df_koersen = df_geldige_koersen().copy()
        print("Min verkoop dt: {}".format(df_koersen["Datum"].min()))

        df_koersen["Eff Interest"] = df_koersen["Datum"].apply(lambda x : equiv_jaar_interest_voor_verkoop(df_koersen, df_mnd_stortingen(), x, input.in_instapkost_pct(), input.in_verkoop_sper_periode()))

        return df_koersen

    @reactive.Calc
    def equiv_jaar_interest():
        req(not df_stortingen_voor_verkoop().empty)

        return bereken_equiv_jaar_interest(df_stortingen_voor_verkoop(), 50)


    @reactive.Calc
    def efficientie():
        req(not df_stortingen_voor_verkoop().empty, input.in_bedrag_storting())

        df = df_stortingen_voor_verkoop()

        inbreng = input.in_bedrag_storting() * df.shape[0]
        verkoop_totaal = df["Verkoop Waarde"].sum()
        winst_pct = 100 * (verkoop_totaal - inbreng) / inbreng
        jaar_interest_pct = equiv_jaar_interest()

        return {
            "inbreng": inbreng,
            "verkoop_totaal": verkoop_totaal,
            "winst_pct": winst_pct,
            "equiv_jaar_interest": jaar_interest_pct
        }

    @reactive.Calc
    def df_verkoop_display():
        req(not df_stortingen_voor_verkoop().empty)

        df = df_stortingen_voor_verkoop().copy()

        df = df.set_index("Datum")

        return df[["Koers", "Aandelen", "Verkoop Waarde", "Dagen Actief"]]


    @reactive.Calc
    def df_stortingen_verkoop():
        df_stortingen_verkoop = pd.merge(df_mnd_stortingen()["Datum"], df_mnd_verkoop_datums()["Datum"], how="cross",
                                         suffixes=("_Eerste_Storting", "_Verkoop"))
        df_stortingen_verkoop = df_stortingen_verkoop[df_stortingen_verkoop["Datum_Verkoop"] > df_stortingen_verkoop["Datum_Eerste_Storting"]]
        df_stortingen_verkoop["Eff Interest"] = df_stortingen_verkoop\
            .apply(lambda x: ji(data(),
                                df_mnd_stortingen(),
                                x["Datum_Eerste_Storting"],
                                x["Datum_Verkoop"],
                                input.in_instapkost_pct(),
                                input.in_verkoop_sper_periode(),
                                input.in_min_stortingen_voor_verkoop()), axis=1)

        df_stortingen_verkoop = df_stortingen_verkoop[df_stortingen_verkoop["Eff Interest"].abs() < 15]

        return df_stortingen_verkoop


    @output
    @render_widget
    def out_eff_interest_eenmalig():
        req(not data().empty, not df_eff_interest().empty, input.in_verkoop_sper_periode())

        df_koers = data()
        df_int = df_eff_interest()
        df_int = df_int[df_int["Datum"] <= verkoop_dt() - timedelta(days=input.in_verkoop_sper_periode())]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_int["Datum"], y=df_int["Eff Interest"], name="Eff Interest", yaxis="y1",
                                 hovertext=df_int["Dagen Actief"].apply(lambda x: "Days active: {}".format(x))))
        fig.add_trace(go.Scatter(x=df_koers["Datum"], y=df_koers["Koers"], line=dict(color="lightgray"), name="Koers", yaxis="y2"))
        fig.add_vline(x=verkoop_dt(), line_width=2, line_color="lime")

        fig.update_layout(
            # create first Y-axis
            yaxis=dict(
                title="Eff Interest (%)"
            ),

            # create second Y-axis
            yaxis2=dict(
                title="Koers (€)",
                overlaying="y",
                side="right",
                position=1
            )
        )
        fig.update_layout(height=600)
        fig.layout.hovermode="closest"

        fig = go.FigureWidget(fig)

        # set selection handler
        # fig.data[0].on_click(click_fn)
        fig.data[1].on_click(click_fn)

        return fig

    def click_fn(trace, points, selector):
        print("Clicked")
        print("Datum Selectie {}".format(points.xs[0]))

        verkoop_dt_doel.set(datetime.strptime(points.xs[0], "%Y-%m-%d"))



    @output
    @render_widget
    def out_eff_interest_maandelijks():
        req(not data().empty, not df_stortingen_voor_verkoop().empty, not df_koersen_met_equiv_jaar_interest().empty)

        df_koers = data()
        df_verk = df_stortingen_voor_verkoop()
        df_ji = df_koersen_met_equiv_jaar_interest()
        df_ji_pos = df_ji[df_ji["Eff Interest"] > 0]
        df_ji_neg = df_ji[df_ji["Eff Interest"] <= 0]
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_koers["Datum"], y=df_koers["Koers"], line=dict(color="lightgray"), name="Koers", yaxis="y2"))
        fig.add_trace(go.Scatter(x=df_verk["Datum"], y=df_verk["Koers"], mode="markers", marker=dict(color="lime"), name="Koers", yaxis="y2"))
        fig.add_trace(go.Scatter(x=df_ji["Datum"], y=df_ji["Eff Interest"], line=dict(color="blue"), name="Eff Interest", yaxis="y1"))

        fig.add_vline(x=verkoop_dt(), line_width=3, line_dash="dash", line_color="lime")

        fig.update_layout(
            # create first Y-axis
            yaxis=dict(
                title="Eff Interest (%)"
            ),

            # create second Y-axis
            yaxis2=dict(
                title="Koers (€)",
                overlaying="y",
                side="right",
                position=1
            )
        )
        fig.update_layout(height=600)
        fig.layout.hovermode="closest"

        fig = go.FigureWidget(fig)

        # set selection handler
        scatter = fig.data[0]
        scatter.on_click(click_fn)

        return fig

    @output
    @render_widget
    def out_maandelijks_data():
        req(not df_verkoop_display().empty)

        w = qgrid.show_grid(df_verkoop_display(), show_toolbar=False)

        return w


    @output
    @render.ui
    def out_samenvatting_maandelijks():
        print(efficientie())
        return ui.tags.table(
            ui.tags.tr(
                ui.tags.th("Verkoop Datum"),
                ui.tags.th("Verkoop Koers"),
                ui.tags.th("Inbreng"),
                ui.tags.th("Verkoop Totaal"),
                ui.tags.th("Winst Pct"),
                ui.tags.th("Equiv. Jaar Interest")
            ),
            ui.tags.tr(
                ui.tags.td(verkoop_dt().strftime("%Y-%m-%d")),
                ui.tags.td(verkoop_koers()),
                ui.tags.td(efficientie()["inbreng"]),
                ui.tags.td(round(efficientie()["verkoop_totaal"], 2)),
                ui.tags.td(round(efficientie()["winst_pct"], 2)),
                ui.tags.td(round(efficientie()["equiv_jaar_interest"], 2)),
            )
        )

    @output
    @render_widget
    def out_eff_interest_matrix():
        req(not df_fonds_matrix().empty)

        dff = df_fonds_matrix()

        n_dt_stortingen = int(dff["Datum_Eerste_Storting"].drop_duplicates().count())
        n_dt_verkoop = int(dff["Datum_Verkoop"].drop_duplicates().count())

        fig = px.density_heatmap(dff,
                                 x="Datum_Eerste_Storting",
                                 y="Datum_Verkoop",
                                 z="Eff Interest",
                                 nbinsx=n_dt_stortingen,
                                 nbinsy=n_dt_verkoop,
                                 histfunc="avg",
                                 color_continuous_midpoint=0,
                                 height=1000)
        fig.update_traces(dict(colorscale=["red", "white", "lime"], showscale=True, coloraxis=None), )
        fig = go.FigureWidget(fig)

        return fig

    @output
    @render_widget
    def out_eff_interest_density():
        req(not df_fonds_matrix().empty)

        dff = df_fonds_matrix()

        min_year = dff["Jaar Eerste Storting"].min()
        max_year = dff["Jaar Eerste Storting"].max()

        jaar_range = range(min_year, max_year + 1)
        jaar_list = [dff["Eff Interest"][dff["Jaar Eerste Storting"] == jaar] for jaar in jaar_range]

        fig = ff.create_distplot(jaar_list, jaar_range, show_hist=False)
        fig.update_layout(height=600)

        fig = go.FigureWidget(fig)

        return fig

    @output
    @render_widget
    def out_koers_vergelijking():
        req(not koers_data_sel().empty)

        df_koers = koers_data_sel()

        fig = px.line(df_koers, x="Datum", y="Relatieve Koers", color="Fonds Naam")

        fig.update_layout(
            # create first Y-axis
            yaxis=dict(
                title="Eff Interest (%)"
            )
        )
        fig.update_layout(height=600)
        fig.layout.hovermode="closest"

        fig = go.FigureWidget(fig)

        return fig

    @output
    @render_widget
    def out_koers_delta_vergelijking():
        req(not koers_data_sel().empty)

        df_koers = koers_data_sel()

        fig = go.Figure()
        fondsen = [f for f in input.in_fonds_vgl()]
        fonds_namen = [df_fondsen.loc[f, "name"] for f in fondsen]

        for i, f in enumerate(fondsen):
            df_koers_fonds = df_koers[df_koers["Fonds"] == f]


            fig.add_trace(go.Scatter(
                x=df_koers_fonds["Datum"],
                y=df_koers_fonds["Koers Delta Min"],
                mode='lines',
                fill=None,
                line_color=px.colors.qualitative.Plotly[i]
            ))
            fig.add_trace(go.Scatter(
                x=df_koers_fonds["Datum"],
                y=df_koers_fonds["Koers Delta Max"],
                mode='lines',
                fill="tonexty",
                line_color=px.colors.qualitative.Plotly[i],
                # fill_color=px.colors.qualitative.Plotly[i],
                name=fonds_namen[i]
            ))

        fig.update_layout(height=600)
        fig.layout.hovermode="closest"

        fig = go.FigureWidget(fig)

        return fig

    @output
    @render_widget
    def out_eff_interest_vergelijking():
        req(input.in_fonds_vgl())

        fondsen = [f for f in input.in_fonds_vgl()]
        fonds_namen = [df_fondsen.loc[f, "name"] for f in fondsen]

        print(fonds_namen)

        print("Vergelijking voor fondsen {}".format(fondsen))
        dfd = df_eff_jaar_interest[df_eff_jaar_interest["Fonds"].isin(fondsen)]

        min_year_st = dfd.groupby("Fonds")["Jaar Eerste Storting"].min().max()
        max_year_st = dfd["Jaar Eerste Storting"].max()
        min_year_v = dfd.groupby("Fonds")["Jaar Verkoop"].min().max()
        max_year_v = dfd["Jaar Verkoop"].max()

        year_list = [dfd["Eff Interest"][dfd["Jaar Eerste Storting"] == jaar] for jaar in
                     range(min_year_st, max_year_st + 1)]

        years_v = range(min_year_v, max_year_v + 1)
        i_v = range(max_year_v - min_year_v + 1)

        years_st = range(min_year_st, max_year_st + 1)
        i_st = range(max_year_st - min_year_st + 1)

        rows = len(years_v)
        cols = len(years_st)

        print(years_v, years_st, i_v, i_st)

        titles = [["" for i in range(cols)] for j in range(rows)]
        figs = [[None for i in range(cols)] for j in range(rows)]

        for r in i_v:
            yr_v = years_v[r]

            for c in i_st:
                # print("row {}, col {}".format(r, c))
                yr_st = years_st[c]
                # print("yr_st {}, yr_v {}".format(yr_st, yr_v))
                if yr_st <= yr_v:

                    df_st_v = [dfd["Eff Interest"][(dfd["Jaar Eerste Storting"] == yr_st) &
                                                      (dfd["Jaar Verkoop"] == yr_v) &
                                                      (dfd["Fonds"] == fonds)].dropna() for fonds in fondsen]

                    min_len = min([len(df_st_v[i].index) for i in range(len(fondsen))])
                    max_len = max([len(df_st_v[i].index) for i in range(len(fondsen))])

                    if min_len > 0:
                        data = [0 for i in fondsen]

                        for i in range(len(data)):
                            if len(df_st_v[i]) < max_len:
                                try:
                                    data[i] = choices(df_st_v[i].tolist(), k=max_len)
                                except:
                                    print(f"error for {yr_st}, {yr_v}, {max_len}, {i}, {len(df_st_v[i])}")
                                    continue
                            else:
                                data[i] = df_st_v[i].tolist()

                        distplot = ff.create_distplot(data, fonds_namen, show_hist=False, show_rug=False)
                        distplot.update_layout(title="({},{})".format(yr_st, yr_v))

                        figs[r][c] = distplot
                        titles[r][c] = "{}-{}".format(yr_st, yr_v)
                    else:
                        figs[r][c] = None
                        titles[r][c] = ""
                else:
                    figs[r][c] = None
                    titles[r][c] = ""

        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[titles[r][c] for r in reversed(range(rows)) for c in range(cols)])

        for r in i_v:
            for c in i_st:
                if figs[r][c]:
                    # print("{}, {}".format(r, c))
                    for i in range(len(fondsen)):
                        fig.add_trace(go.Scatter(figs[r][c].data[i], showlegend=(r + c == 0)), row=rows - r, col=c + 1)

        fig.update_layout(
            height=1600,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.5
            )
        )

        return go.FigureWidget(fig)


app = App(app_ui, server)

def main():
    run_app(app)

if __name__ == "__main__":
    main()