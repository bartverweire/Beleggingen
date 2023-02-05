class Input():
    in_bedrag_storting = 100
    in_instapkost_pct = 3
    in_verkoop_sper_periode = 90
    in_min_stortingen_voor_verkoop = 2
    in_dag_storting = 10
    in_dag_verkoop = 20

    def __init__(self,
                 _in_bedrag_storting=100,
                 _in_instapkost_pct=3,
                 _in_verkoop_sper_periode=90,
                 _in_min_stortingen_voor_verkoop=2,
                 _in_dag_storting=10,
                 _in_dag_verkoop=20
    ):
        self._in_bedrag_storting = _in_bedrag_storting
        self._in_instapkost_pct = _in_instapkost_pct
        self._in_verkoop_sper_periode = _in_verkoop_sper_periode
        self._in_min_stortingen_voor_verkoop = _in_min_stortingen_voor_verkoop
        self._in_dag_storting = _in_dag_storting
        self._in_dag_verkoop  = _in_dag_verkoop


    def in_bedrag_storting(self):
        return self._in_bedrag_storting

    def in_instapkost_pct(self):
        return self._in_instapkost_pct

    def in_verkoop_sper_periode(self):
        return self._in_verkoop_sper_periode

    def in_min_stortingen_voor_verkoop(self):
        return self._in_min_stortingen_voor_verkoop

    def in_dag_storting(self):
        return self._in_dag_storting

    def in_dag_verkoop(self):
        return self._in_dag_verkoop
