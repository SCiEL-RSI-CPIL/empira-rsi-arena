"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
INPUT_TYPE = "structure"
CLASSIFICATION = True
# EVOLVE-BLOCK-START
import numpy as np

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru "
            "Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
            "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
INDEX = {symbol: k for k, symbol in enumerate(ELEMENTS)}


def features(x):
    """Element fractions plus mean, spread and extremes of the atomic number; structures add volume per atom."""
    if INPUT_TYPE == "composition":
        amounts = {INDEX[e]: float(a) for e, a in x["composition"].items() if e in INDEX}
    else:
        amounts = {}
        for symbol in x["species"]:
            amounts[INDEX[symbol]] = amounts.get(INDEX[symbol], 0.0) + 1.0
    total = sum(amounts.values())
    fractions = np.zeros(len(ELEMENTS))
    z = np.array(list(amounts)) + 1.0
    weight = np.array(list(amounts.values())) / total
    for k, a in amounts.items():
        fractions[k] = a / total
    mean = float((z * weight).sum())
    extra = [mean, float(np.sqrt((weight * (z - mean) ** 2).sum())), float(z.min()), float(z.max()), float(len(amounts))]
    if INPUT_TYPE == "structure":
        volume = abs(np.linalg.det(np.asarray(x["lattice"], dtype=float)))
        extra += [volume / len(x["species"]), float(len(x["species"]))]
    return np.concatenate([fractions, extra])


def fit_predict(train_inputs, train_targets, test_inputs, info):
    """Random forest on element fractions and simple statistics (baseline)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X = np.array([features(x) for x in train_inputs])
    Xt = np.array([features(x) for x in test_inputs])
    y = np.asarray(train_targets, dtype=float)
    trees = 50 if len(X) > 20000 else 200
    if CLASSIFICATION:
        model = RandomForestClassifier(n_estimators=trees, n_jobs=4, random_state=0).fit(X, y > 0.5)
        return [float(v) for v in model.predict(Xt)]
    model = RandomForestRegressor(n_estimators=trees, n_jobs=4, random_state=0).fit(X, y)
    return model.predict(Xt).tolist()
# EVOLVE-BLOCK-END
