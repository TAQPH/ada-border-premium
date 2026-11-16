"""
Stage 1 - obtain the raw data. Run from the repository root:

    python code/01_get_data.py

Downloads into data/raw/ :
  dvf/dvf_74_<year>.csv.gz        geolocated DVF sales, Haute-Savoie, one file per year
  insee/filosofi_2021_74.csv      income by municipality (INSEE Filosofi 2021)
  insee/logement_2021_74.csv      housing stock by municipality (INSEE census 2021),
                                  used for the share of second homes
  insee/mobpro_74.csv             commuting records of Haute-Savoie residents
                                  (INSEE census 2021), used for the share of
                                  cross-border workers

Files already present are not downloaded again. A failed download does not
stop the script: a summary is printed at the end.

Sources and licences
  DVF geolocalisees (Etalab / DGFiP), Licence Ouverte 2.0
    https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees
  INSEE Filosofi 2021, open data  https://www.insee.fr/fr/statistiques/7756855
  INSEE census 2021, housing by municipality, open data
    https://www.insee.fr/fr/statistiques/8202349
  INSEE census 2021, "mobilites professionnelles", open data
    https://www.insee.fr/fr/statistiques/8205896

Tool declaration: drafted with Claude (Anthropic), asked to write a download
script for these sources.
"""
import shutil
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from helpers import DEPARTMENT, RAW, YEARS

DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/{dep}.csv.gz"
FILOSOFI_URL = ("https://www.insee.fr/fr/statistiques/fichier/7756855/"
                "indic-struct-distrib-revenu-2021-COMMUNES_csv.zip")
HOUSING_URL = ("https://www.insee.fr/fr/statistiques/fichier/8202349/"
               "base-cc-logement-2021_csv.zip")
MOBPRO_URL = "https://www.insee.fr/fr/statistiques/fichier/8205896/RP2021_mobpro.parquet"

summary = []  # (label, status) pairs printed at the end


def download(url, target, label):
    """Download url to target unless it already exists. Returns True if the file is there."""
    if target.exists():
        summary.append((label, "already present"))
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as f:
            shutil.copyfileobj(response, f)
    except (urllib.error.URLError, TimeoutError) as error:
        target.unlink(missing_ok=True)  # do not leave a half-written file
        summary.append((label, f"FAILED ({error})"))
        return False
    summary.append((label, f"downloaded ({target.stat().st_size / 1e6:.1f} MB)"))
    return True


# 1. DVF: one file per year for the department
for year in YEARS:
    download(DVF_URL.format(year=year, dep=DEPARTMENT),
             RAW / "dvf" / f"dvf_{DEPARTMENT}_{year}.csv.gz",
             f"DVF {year}")

# 2. Filosofi: national zip (about 17 MB), we keep only the municipalities of Haute-Savoie
filosofi_zip = RAW / "insee" / "filosofi_2021_communes.zip"
filosofi_dep = RAW / "insee" / f"filosofi_2021_{DEPARTMENT}.csv"
if filosofi_dep.exists():
    summary.append(("Filosofi 2021", "already present"))
elif download(FILOSOFI_URL, filosofi_zip, "Filosofi 2021"):
    with zipfile.ZipFile(filosofi_zip) as z, z.open("FILO2021_DISP_COM.csv") as f:
        filosofi = pd.read_csv(f, sep=";", dtype=str)  # disposable income by municipality
    keep = filosofi["CODGEO"].str.startswith(DEPARTMENT)
    filosofi[keep].to_csv(filosofi_dep, index=False)
    print(f"Kept {keep.sum()} of {len(filosofi):,} municipalities (Filosofi).")

# 3. Census housing file: national zip (about 39 MB), we keep Haute-Savoie and four columns
housing_zip = RAW / "insee" / "logement_2021_communes.zip"
housing_dep = RAW / "insee" / f"logement_2021_{DEPARTMENT}.csv"
housing_columns = ["CODGEO", "P21_LOG", "P21_RP", "P21_RSECOCC"]  # all homes, main homes, second homes
if housing_dep.exists():
    summary.append(("Census housing file", "already present"))
elif download(HOUSING_URL, housing_zip, "Census housing file"):
    with zipfile.ZipFile(housing_zip) as z:
        # the zip holds the data and a small "meta" file: the data is the largest one
        data_name = max(z.infolist(), key=lambda member: member.file_size).filename
        with z.open(data_name) as f:
            housing = pd.read_csv(f, sep=";", dtype=str)
    if set(housing_columns) <= set(housing.columns):
        keep = housing["CODGEO"].str.startswith(DEPARTMENT)
        housing.loc[keep, housing_columns].to_csv(housing_dep, index=False)
        print(f"Kept {keep.sum()} of {len(housing):,} municipalities (housing).")
    else:
        print(f"Expected columns not found in {data_name}. First columns are:", list(housing.columns)[:15])

# 4. Census commuting file: national (63 MB), we keep only Haute-Savoie residents
mobpro_national = RAW / "insee" / "RP2021_mobpro.parquet"
mobpro_dep = RAW / "insee" / f"mobpro_{DEPARTMENT}.csv"
if mobpro_dep.exists():
    summary.append(("Census commuting file", "already present"))
elif download(MOBPRO_URL, mobpro_national, "Census commuting file"):
    mobpro = pd.read_parquet(mobpro_national)
    if "COMMUNE" in mobpro.columns:  # COMMUNE = municipality of residence
        keep = mobpro["COMMUNE"].astype(str).str.startswith(DEPARTMENT)
        mobpro[keep].to_csv(mobpro_dep, index=False)
        print(f"Kept {keep.sum():,} of {len(mobpro):,} commuting records (residents of {DEPARTMENT}).")
    else:
        print("Column COMMUNE not found. Columns are:", list(mobpro.columns))

# Summary
print("\n=== Summary ===")
for label, status in summary:
    print(f"{label:25s} {status}")

print("\n=== Files in data/raw ===")
for path in sorted(RAW.rglob("*")):
    if path.is_file():
        print(f"{path.relative_to(RAW).as_posix():55s} {path.stat().st_size / 1e6:8.1f} MB")
