# Verkefni 2 – Price-and-Cut

Verkefni í Aðgerðagreiningu 2 við Háskóla Íslands.

Verkefnið snýst um að leysa skurðarvandann (*Cutting Stock Problem*) með dálkamyndun (*Column Generation*) og Gomory-skurðum.

## Python-skrár

- `price_and_cut.py` – Aðalreikniritið.
- `starter_code.py` – Hjálparföll sem aðalreikniritið notar.
- `verify.py` – Óháð sannprófun á niðurstöðum.
- `degeneracy_test.py` – Prófun á úrkynjun og hringrás (einstaklingsverkefni).
- `staerdarprof.py` – Stærðar- og hraðaprófanir.

## Keyrsla

Keyra skal eftirfarandi skipanir úr `Verkefni-2` möppunni.

**Sannprófun aðalreiknirits:**

```bash
python verify.py
```

**Úrkynjunarpróf:**

```bash
python degeneracy_test.py
```

**Stærðar- og hraðaprófanir:**

```bash
python staerdarprof.py
```

## Kröfur

- Python 3
- `matplotlib` fyrir stærðarprófanir
- `scipy` fyrir valfrjálsa krossprófun í `verify.py`

Gagnasettin eru skilgreind í Python-skránum. Ekki þarf Gurobi-leyfi eða utanaðkomandi gagnaskrár.

Allar fimm Python-skrárnar þurfa að vera í sömu möppu.
