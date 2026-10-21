# Projekto reikalavimai

**Versija:** 1.1
**Data:** 2026-10-21
**Autorius:** Matas

## 1. Problemos aprašymas

Reprodukuojamas polinominės regresijos MLOps pipeline.

## 2. Funkciniai reikalavimai (FR)

| ID | Reikalavimas | Prioritetas |
|----|--------------|-------------|
| FR-01 | Sistema generuoja polinominius duomenis pagal `params.yaml` į `data/raw/data.csv`. | Must |
| FR-02 | Sistema treniruoja `PolynomialFeatures + LinearRegression` ir log'ina parametrus, metrikas (`MSE`, `R2`) bei modelį į MLflow. | Must |
| FR-03 | Sistema atkuria istorinį MLflow run'ą pagal `run_id` (nurodytą `models/run_id.txt`): perrašo `params.yaml`, paleidžia `dvc repro --force`, log'ina naują run'ą į MLflow. | Must |

## 3. Nefunkciniai reikalavimai (NFR)

| ID | Kategorija | Reikalavimas | KPI |
|----|------------|--------------|-----|
| NFR-01 | Reprodukcija (parametrai) | `python src/reproduce.py` teisingai atkuria originalaus run'o parametrus | Parametrai naudojami reprodukcijai lygūs originaliems run'o parametrams |
| NFR-02 | Reprodukcija (metrikos) | Atkurto run'o eval metrikos beveik vienodos originaliam run'ui | ΔR² ≤ 0.005; ΔMSE ≤ 1% |
| NFR-03 | Našumas | Visas pipeline'as (prepare + train + evaluate) | Vykdymas < 15 s |
| NFR-04 | Docker dydis | Image dydis | < 700 MB |
| NFR-05 | Duomenų išsaugojimas | Run'ai išlieka po Pod restart | 100% run'ų išlieka |