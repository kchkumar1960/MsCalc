# coding: utf-8
# Web deployment of Ms predictor (CatBoostRegressor)
import sys
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

el = [
    "C",
    "Mn",
    "Si",
    "Cr",
    "Ni",
    "Mo",
    "V",
    "Co",
    "Al",
    "W",
    "Cu",
    "Nb",
    "Ti",
    "B",
    "N",
    "P",
    "S"
]

# List of element symbols
el = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo','V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti',
      'B', 'N', 'P', 'S']

n = len(el)

wp = [0.0] * n

# Loop through each element and prompt the user input composition
for i in range(n):
    user_input = input(f"Input mass% of {el[i]} [0.0]: ").strip()
    if user_input:  # if user typed something
        try:
            wp[i] = abs(float(user_input))  # convert to float
        except ValueError:
            print("Invalid input, keeping default 0.0.")


if sum(wp) > 100.0:
    print("Total exceeds 100.0, exiting program.")
    sys.exit()

wp = [wp]

df = pd.DataFrame(wp, columns=el)

atomicmass = {  # [g/mol]
    "Fe": 55.847,
    "C": 12.011,
    "Mn": 54.93805,
    "Si": 28.0855,
    "Cr": 51.9961,
    "Ni": 58.6934,
    "Mo": 95.95,
    "V": 50.9415,
    "Co": 58.9332,
    "Al": 26.981539,
    "W": 183.84,
    "Cu": 63.546,
    "Nb": 92.90638,
    "Ti": 47.867,
    "B": 10.811,
    "N": 14.00674,
    "P": 30.973762,
    "S": 32.066
}

en_el = {  # en of elements (Allen scale)
    "Fe": 1.8,
    "C": 2.544,
    "Mn": 1.75,
    "Si": 1.916,
    "Cr": 1.65,
    "Ni": 1.88,
    "Mo": 1.47,
    "V": 1.53,
    "Co": 1.84,
    "Al": 1.613,
    "W" : 1.47,
    "Cu": 1.85,
    "Nb": 1.41,
    "Ti": 1.38,
    "B": 2.051,
    "N": 3.066,
    "P": 2.253,
    "S": 2.589
}

atomicradius = {  # in pm
    "Fe": 125,
    "C": 77,
    "Mn": 132,
    "Si": 110,
    "Cr": 130,
    "Ni": 128,
    "Mo": 139,
    "V": 134,
    "Co": 125,
    "Al": 141,
    "W": 135,
    "Cu": 127,
    "Nb": 143,
    "Ti": 142,
    "B": 78,
    "N": 72,
    "P": 102,
    "S": 103
}


def wp2mf(composition):
    """
    Convert weight percent composition to mole fractions in a steel alloy.

    Parameters:
    - composition (dict): A dictionary specifying the weight percent
                          composition of the alloy.
                          The keys are element symbols, and the values are the
                          weight percent fractions.

    Returns:
    - mole_fractions (dict): A dictionary containing the mole fractions of each
                             element in the alloy.
                             The keys are element symbols, and the values are
                             the corresponding mole fractions.

    Notes:
    - The function assumes the basis of iron (Fe) as 100% unless explicitly
      specified.
      If an element is not specified in the composition, its weight percent
      fraction is assumed to be 0%.
    - The atomic weights of the elements are based on standard values.

    Raises:
    - ValueError: If an element symbol in the composition is not found in the
                  atomicmass dictionary.
    - AssertionError: If Fe is explicitly specified and the composition does
                      not sum up to 100% (with a tolerance of 0.5%).
    """

    element_compositions = {}
    total_composition = 0.0

    for element, fraction in composition.items():
        if element not in atomicmass:
            raise ValueError(
                f"Atomic mass not found for element: {element}"
            )
        element_compositions[element] = fraction / 100
        total_composition += element_compositions[element]

    if "Fe" not in element_compositions:
        element_compositions["Fe"] = 1.0 - total_composition
        total_composition = 1
    else:
        assert abs(total_composition - 1.0) <= 0.005

    mole_fractions = {}
    mole_sum = 0
    for element, fraction in element_compositions.items():
        mole_fractions[element] = (
            element_compositions[element] / atomicmass[element]
        )
        mole_sum += mole_fractions[element]

    for element, fraction in element_compositions.items():
        mole_fractions[element] = mole_fractions[element] / mole_sum

    return mole_fractions


#  wp -> mf conversion of each row of composition in df
mf_comp_list = []
for index, row in df.iterrows():
    wp_comp = {col: row[col] for col in el}
    mf_comp = wp2mf(wp_comp)
    mf_comp_list.append(mf_comp)

df_mf = pd.DataFrame(mf_comp_list)
col_to_move = df_mf.pop("Fe")
df_mf.insert(0, "Fe", col_to_move)


def lp_ferrite(row):
    """
    Calculates lattice parameter (in pm) of ferrite
    """

    # Bhadhesia-1991, composition in mole fraction, lattice parameter in nm
    aFe = 0.28664
    lp_bcc = aFe + \
        ((aFe-0.0279*row["C"])**2*(aFe+0.2496*row["C"])-aFe**3)/3/aFe**2 \
        - 0.003 * row["Si"] + 0.006 * row["Mn"] + 0.007 * row["Ni"] \
        + 0.031 * row["Mo"] + 0.005 * row["Cr"] + 0.0096 * row["V"]
    return lp_bcc*1000


def lp_austenite(row):
    """
    Calculates lattice parameter (in pm) of austenite (
    """

    wC = row["C"]
    wMn = row["Mn"]
    wNi = row["Ni"]
    wCr = row["Cr"]
    wAl = row["Al"]
    wMo = row["Mo"]
    wV = row["V"]
    wN = row["N"]
    wCo = row["Co"]
    wCu = row["Cu"]
    wNb = row["Nb"]
    wTi = row["Ti"]
    wW = row["W"]

    # Krolicka-2025, composition in wt.%, lattice parameter in Armstrong
    lp_fcc = 3.578 + 0.033 * wC + 0.00095 * wMn - 0.0002 * wNi +\
        0.0006 * wCr + 0.022 * wN + 0.0056 * wAl - 0.0004 * wCo +\
        0.0015 * wCu + 0.0031 * wMo + 0.0051 * wNb + 0.0039 * wTi +\
        0.0018 * wV + 0.0018 * wW
    return lp_fcc*100


def loc_en_mis(row):
    """
    Calculates local electronegativity mismatch (en_lmis)
    """

    n = len(row)
    en_lmis = 0
    for i in range(n):
        for j in range(i+1, n):
            if row.iloc[i] > 0.0 and row.iloc[j] > 0.0:
                en_lmis += row.iloc[i] * row.iloc[j] *\
                    abs(en_el[row.index[i]] - en_el[row.index[j]])
    return 2*en_lmis


def enCrms(row):
    """
    Calculates electronegativity rms mismatch with C (enC_rms)
    """

    enC_rms = sum(row[col] * (1-en_el[col]/en_el["C"])**2
                  for col in row.index if col in en_el)
    return np.sqrt(enC_rms)


def loc_ar_mis(row):
    """
    Calculates local atomic radius mismatch (ar_lmis)
    """

    n = len(row)
    ar_lmis = 0
    for i in range(n):
        for j in range(i+1, n):
            if row.iloc[i] > 0.0 and row.iloc[j] > 0.0:
                ar_lmis += row.iloc[i] * row.iloc[j] *\
                    abs(atomicradius[row.index[i]]
                        - atomicradius[row.index[j]])
    return 2*ar_lmis


def arCrms(row):
    """
    Calculates atomic radius mismatch with C (arC_rms)
    """

    arC_rms = sum(row[col] * (1-atomicradius[col]/atomicradius["C"])**2
                  for col in row.index if col in atomicradius)
    return np.sqrt(arC_rms)


def Ghosh(row):
    """
    Calculates critical driving force according to Ghosh-Olsen model
    """

    ghosh_1 = np.sqrt(
        (4009 * np.sqrt(row["C"]))**2
        + (3097 * np.sqrt(row["N"]))**2
    )
    ghosh_2 = np.sqrt(
        (1868 * np.sqrt(row["Cr"]))**2
        + (1980 * np.sqrt(row["Mn"]))**2
        + (1418 * np.sqrt(row["Mo"]))**2
        + (1653 * np.sqrt(row["Nb"]))**2
        + (1879 * np.sqrt(row["Si"]))**2
        + (1473 * np.sqrt(row["Ti"]))**2
        + (1618 * np.sqrt(row["V"]))**2
    )
    ghosh_3 = np.sqrt(
        (280 * np.sqrt(row["Al"]))**2
        + (752 * np.sqrt(row["Cu"]))**2
        + (172 * np.sqrt(row["Ni"]))**2
        + (714 * np.sqrt(row["W"]))**2
    )
    ghosh_4 = -352 * np.sqrt(row["Co"])

    dGc = 1010 + ghosh_1 + ghosh_2 + ghosh_3 + ghosh_4
    return dGc


def sconfbcc(row):
    """
    Calculates configurational entropy of BCC
    """

    a_1 = 1.0
    a_2 = 3.0
    # Split components into two lists
    sl_2 = ["B", "C", "N"]
    sl_1 = list(filter(lambda x: x not in sl_2, row.index))
    # Create a composition array for elements in sublattice 1 and 2
    x_1 = [row.get(el) for el in sl_1 if row[el] > 0.0]
    x_2 = [row.get(el) for el in sl_2 if row[el] > 0.0]
    if len(x_2) > 0:
        # Calculate site fractions
        y_1 = x_1/(1.0-sum(x_2))
        y_2 = x_2/(1.0-sum(x_2))
        y_2 = y_2 * (a_1/a_2)
        # Append site fraction of vacancies
        y_Va = 1.0 - sum(y_2)
        y_2 = np.append(y_2, y_Va)
        sconf_bcc = a_1 * sum(y_1 * np.log(y_1)) + a_2 * sum(y_2 * np.log(y_2))
        sconf_bcc = sconf_bcc/(a_1 + a_2 * (1-y_Va))
    else:
        sconf_bcc = sum(x_1 * np.log(x_1))

    sconf_bcc = -8.31452 * sconf_bcc
    return sconf_bcc


extra_features = pd.DataFrame()
extra_features["en_lmis"] = df_mf.apply(loc_en_mis, axis=1)
extra_features["enC_rms"] = df_mf.apply(enCrms, axis=1)
extra_features["ar_lmis"] = df_mf.apply(loc_ar_mis, axis=1)
extra_features["arC_rms"] = df_mf.apply(arCrms, axis=1)
extra_features["lp_fcc"] = df.apply(lp_austenite, axis=1)
extra_features["lp_bcc"] = df_mf.apply(lp_ferrite, axis=1)
extra_features["Sconf_bcc"] = df_mf.apply(sconfbcc, axis=1)
extra_features["dGc"] = df_mf.apply(Ghosh, axis=1)

df_mf = df_mf.drop(columns=["Fe"])
df_mf = pd.concat([df_mf, extra_features], axis=1)

# Load scaler
scaler = joblib.load('x_scaler.pkl')

# Scale features
X_alloy_scaled = scaler.transform(df_mf)

# Load trained model with all data (training+unseen)
model = CatBoostRegressor()
model.load_model('catboost_final-r.cbm')

# Predict Ms
Ms_pred = np.round(model.predict(X_alloy_scaled)-273.15, 0)

print("Ms predicted (C) ", Ms_pred.tolist())

