# Verkefni 1 – Stundatöflubestun

Verkefni í Aðgerðagreiningu 2 við Háskóla Íslands.

Markmið verkefnisins er að bæta stundatöflu Verkfræði- og náttúruvísindasviðs HÍ fyrir viku 38 með bestunarlíkani í Python og Gurobi.

## Python-skrár

- `timetable_final.py` – Bestunarlíkan sem lágmarkar vegið árekstraskor milli námskeiða.
- `timetable_balanced.py` – Útvíkkað líkan sem tekur einnig tillit til seinnar kennslu og matarhlés nemenda.

Bæði líkönin nota sömu grunnskorður, meðal annars varðandi stofur, sameiginlegar kennslulotur, fyrsta árs námskeið og matarhlé.

## Gögn

Til að keyra líkönin þarf þrjár CSV-gagnaskrár frá HÍ:

- `Bókanir.csv` – Kennslulotur, tímasetningar og stofur.
- `Kjörsvið.csv` – Flokkun námskeiða eftir námsleiðum og námsárum.
- `Árekstur.csv` – Vægi árekstra milli námskeiða.

Upprunalegu gagnaskrárnar fylgja ekki repository-inu og þarf að nálgast þær sérstaklega.

## Kröfur

- Python 3
- `gurobipy`
- `pandas`
- Gilt Gurobi-leyfi

Til að endurtaka keyrslurnar þarf að hafa CSV-skrárnar tiltækar og stilla gagnaslóð að þeim stað þar sem þær eru vistaðar.

Líkönin voru upphaflega keyrð í Google Colab með gögnum á Google Drive.

**Athugið:** Gurobi-leyfisskrá og leyfisupplýsingar fylgja ekki kóðanum.
