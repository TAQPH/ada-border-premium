# How much is being close to Switzerland worth for house prices?

Victor Menetrier - Advanced Data Analytics, HEC Lausanne, Autumn 2026.

This project measures how much proximity to the Swiss border raises the price
per square metre of homes in Haute-Savoie, and whether this border premium
changed between 2021 and 2025. It uses about 77,000 home sales and compares a
linear regression, a random forest, a gradient boosting and a Gaussian process.

The paper is `paper.pdf`. The recorded presentation: (link to add).

## Install

    pip install -r requirements.txt

## Run, in this order

Step 1 is a script, run from the repository root. Steps 2 to 6 are notebooks:
open each one and run all cells, top to bottom.

| Step | File | What it does | What it produces |
|------|------|--------------|------------------|
| 1 | `python code/01_get_data.py` | downloads the DVF and INSEE data | files in `data/raw/` |
| 2 | `code/02_clean.ipynb` | turns DVF into one row per home sold | `data/processed/sales_clean.csv`, `cleaning_log.csv` |
| 3 | `code/03_features.ipynb` | adds the distance to the border and the municipality data | `data/processed/sales_features.csv` |
| 4 | `code/04_models.ipynb` | linear regression, random forest, gradient boosting; border effect by year | `model_comparison.csv`, `boosting_predictions.csv`, `border_shape.csv`, `border_effect_by_year.csv` in `data/processed/` |
| 5 | `code/05_gaussian_process.ipynb` | Gaussian process on the model residuals: map of unexplained prices | `gp_surface.csv`, `gp_by_distance.csv` in `data/processed/` |
| 6 | `code/06_figures.ipynb` | draws every figure and table of the paper | the five figures in `figures/` |

`code/helpers.py` holds the functions shared by several steps.

The raw data is included in the repository, so step 1 only checks that the
files are there and downloads nothing. Steps 2 to 6 can be run directly.

Running time: about 5 minutes in total on a laptop (3 minutes for step 4, under
1 minute for each other step). No resource beyond the Nuvolos workspace is needed.

## Data

All sources are open data and may be redistributed with attribution.

- DVF geolocalisees (Etalab / DGFiP), Licence Ouverte 2.0:
  https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees
  Sales in Haute-Savoie, 2021 to 2025. The source only keeps the last five
  years, which is why the files are stored here: `data/raw/dvf/`.
- INSEE Filosofi 2021, income by municipality: https://www.insee.fr/fr/statistiques/7756855
- INSEE census 2021, housing by municipality: https://www.insee.fr/fr/statistiques/8202349
- INSEE census 2021, commuting file: https://www.insee.fr/fr/statistiques/8205896
  The three INSEE files are reduced to Haute-Savoie in `data/raw/insee/`
  (INSEE data: free reuse with mention of the source).
- `data/raw/border_crossings.csv`: 16 road border crossings between
  Haute-Savoie and Switzerland, compiled by hand; coordinates are accurate to
  a few hundred metres (checked in `03_features.ipynb`).

## Tools

The code was drafted with Claude (Anthropic); each file states what the tool
was asked to produce. The full declaration is in the appendix of the paper.
