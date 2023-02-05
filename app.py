from shiny import *
from shinywidgets import *
from datetime import datetime, date, timedelta
from os import path


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import qgrid

from calc import *

df_fondsen = pd.read_csv("fondsen.csv")
df_fondsen = df_fondsen.dropna()
df_fondsen = df_fondsen.set_index("id")

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
                # ui.nav(
                #     "Maandelijkse Storting",
                #     ui.h4("Effective Interest"),
                #     output_widget("out_eff_interest_maandelijks"),
                #     ui.output_ui("out_samenvatting_maandelijks"),
                #     output_widget("out_maandelijks_data")
                # ),
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

app = App(app_ui, server)

def main():
    run_app(app)

if __name__ == "__main__":
    main()