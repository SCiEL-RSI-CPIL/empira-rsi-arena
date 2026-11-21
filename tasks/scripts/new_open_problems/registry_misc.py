"""Records for the smaller families, first read from public tables on 2026-11-02.

scripts/validate_records.py checks every value against pinned sources. The QAP values were replaced on 2026-11-15 by those of
the maintained QAPLIB table; the snake-in-the-box records are not stored here (the validation reads them from the leaderboard).
"""

# ---- QAPLIB, instances not solved to optimality: name -> (n, best known, best lower bound). tai256c, tho150, wil100 are
# ---- already tasks of autoresearch-tasks and are left out. Values of the maintained QAPLIB table (qaplib.mgi.polymtl.ca, read in the
# ---- Internet Archive copy of 2026-05-09, unchanged since 2020), compared on 2026-11-15: it has stronger lower bounds than the first version
# ---- for 18 instances. One best known value is not QAPLIB's: tai100a, 21043560 (Misevicius 2019, Memetic Computing 11:331-332), where QAPLIB
# ---- still lists 21044752 (2017) and the first version had 21052466 from an older mirror. scripts/validate_records.py checks all of this.
QAP = {
    "sko42": (42, 15812, 15332), "sko49": (49, 23386, 22650), "sko56": (56, 34458, 33385), "sko64": (64, 48498, 47017),
    "sko72": (72, 66256, 64455), "sko81": (81, 90998, 88359), "sko90": (90, 115534, 112423), "sko100a": (100, 152002, 143846),
    "sko100b": (100, 153890, 145522), "sko100c": (100, 147862, 139881), "sko100d": (100, 149576, 141289), "sko100e": (100, 149150, 140893),
    "sko100f": (100, 149036, 140691), "tai30a": (30, 1818146, 1706855), "tai35a": (35, 2422002, 2216627), "tai35b": (35, 283315445, 269532400),
    "tai40a": (40, 3139370, 2843274), "tai40b": (40, 637250948, 608808400), "tai50a": (50, 4938796, 4390920),
    "tai50b": (50, 458821517, 431090700), "tai60a": (60, 7205962, 6325978), "tai60b": (60, 608215054, 592371800),
    "tai80a": (80, 13499184, 11657010), "tai80b": (80, 818415043, 786298800), "tai100a": (100, 21040996, 17853840),
    "tai100b": (100, 1185996137, 1151591000), "tai150b": (150, 498896643, 441786736), "tho40": (40, 240516, 226490),
    "wil50": (50, 48816, 48121),
}
# Instances whose best published value is later than QAPLIB's: the value, QAPLIB's own solution file (replayed), and the paper the value is read from.
QAP_LATER = {"tai100a": {
    "value": 21040996,
    "qaplib_solution": ("http://web.archive.org/web/20200812120955id_/http://anjos.mgi.polymtl.ca:80/qaplib/soln.d/tai100a.sln", "qaplib_polymtl_tai100a.sln"),
    "permutation_file": "scripts/qap_tai100a_2024.json",   # the paper prints the permutation; the validation replays it
    "paper_cite": ("Sergienko, Shylo, Roshchyn, Shylo, Boyarchuk and Moroz 2024 (Cybernetics and Computer Technologies 2024(3):25-33, doi:10.34229/2707-451X.24.3.3), "
                   "which prints the permutation; earlier values were 21,044,752 (V. Shylo 2017, QAPLIB) and 21,043,560 (Misevicius 2019, Memetic Computing 11:331-332)"),
    "unverified": ("no newer tai100a value was found: the 2026 paper that an earlier version of this review cited as holding a better one (Sergienko, Shylo, Roshchyn and Boyarchuk, "
                   "Cybernetics and Systems Analysis 62(4):578-587, doi 10.1007/s10559-026-00893-0) is titled 'Applying Algorithm Unions to Specific Classes of Boolean Programming Problems' (Crossref), "
                   "so it is not about this instance"),
}}

assert len(QAP) == 29

# ---- Job shop, open instances: name -> (jobs, machines, best lower bound, best known upper bound).
# ---- Bounds as of 2026-11-11, merged from two maintained tables: O. Shylo's bounds pages (optimizizer.com/TA.php, DMU.php, Extra.php, which also
# ---- publish a schedule for every upper bound) and the ScheduleOpt job-shop benchmark library (github.com/ScheduleOpt/benchmarks, jobshop/README.md).
# ---- The lower bound is the larger of the two, the upper bound the smaller. The first version of this table was read from
# ---- thomasWeise/jsspInstancesAndResults, which is no longer updated: 25 of its 85 "open" instances are closed today (JSSP_CLOSED), and 40 of
# ---- the other 60 have a better upper bound. scripts/validate_records.py --families jssp re-derives every row from the pinned snapshots.
JSSP = {}
def _j(prefix, rows):
    for name, jobs, machines, lb, ub in rows:
        JSSP[name] = (jobs, machines, lb, ub)
_j("ta", [("ta30", 20, 20, 1562, 1584), ("ta32", 30, 15, 1774, 1783), ("ta40", 30, 15, 1658, 1669), ("ta41", 30, 20, 1926, 2005),
          ("ta42", 30, 20, 1900, 1937), ("ta43", 30, 20, 1809, 1846), ("ta44", 30, 20, 1961, 1978), ("ta46", 30, 20, 1976, 2002),
          ("ta47", 30, 20, 1827, 1889), ("ta48", 30, 20, 1921, 1937), ("ta49", 30, 20, 1938, 1960), ("ta50", 30, 20, 1848, 1923)])
_j("dmu", [("dmu11", 30, 15, 3402, 3430), ("dmu12", 30, 15, 3481, 3492), ("dmu16", 30, 20, 3734, 3744),
           ("dmu17", 30, 20, 3733, 3811), ("dmu19", 30, 20, 3707, 3764), ("dmu20", 30, 20, 3632, 3699),
           ("dmu41", 20, 15, 3176, 3248), ("dmu42", 20, 15, 3339, 3390), ("dmu44", 20, 15, 3414, 3475),
           ("dmu45", 20, 15, 3217, 3266), ("dmu46", 20, 20, 3780, 4035), ("dmu47", 20, 20, 3714, 3939),
           ("dmu48", 20, 20, 3628, 3763), ("dmu49", 20, 20, 3543, 3706), ("dmu50", 20, 20, 3618, 3729),
           ("dmu51", 30, 15, 4070, 4151), ("dmu52", 30, 15, 4203, 4297), ("dmu53", 30, 15, 4248, 4378),
           ("dmu54", 30, 15, 4277, 4360), ("dmu55", 30, 15, 4191, 4258), ("dmu56", 30, 20, 4755, 4934),
           ("dmu57", 30, 20, 4462, 4643), ("dmu58", 30, 20, 4484, 4701), ("dmu59", 30, 20, 4366, 4607),
           ("dmu60", 30, 20, 4468, 4721), ("dmu61", 40, 15, 5038, 5159), ("dmu62", 40, 15, 5176, 5244),
           ("dmu63", 40, 15, 5245, 5296), ("dmu64", 40, 15, 5155, 5225), ("dmu65", 40, 15, 5122, 5158),
           ("dmu66", 40, 20, 5526, 5690), ("dmu67", 40, 20, 5661, 5774), ("dmu68", 40, 20, 5513, 5735),
           ("dmu69", 40, 20, 5511, 5682), ("dmu70", 40, 20, 5633, 5868), ("dmu71", 50, 15, 6129, 6204),
           ("dmu72", 50, 15, 6434, 6448), ("dmu73", 50, 15, 6107, 6132), ("dmu74", 50, 15, 6168, 6196),
           ("dmu75", 50, 15, 6123, 6186), ("dmu76", 50, 20, 6479, 6708), ("dmu77", 50, 20, 6520, 6739),
           ("dmu78", 50, 20, 6643, 6744), ("dmu79", 50, 20, 6720, 6899), ("dmu80", 50, 20, 6460, 6621)])
_j("swv", [("swv07", 20, 15, 1541, 1594), ("swv08", 20, 15, 1694, 1751), ("swv10", 20, 15, 1692, 1743)])
assert len(JSSP) == 60

# ---- Instances of the first table that are solved today (lower bound = upper bound in the merged tables): name -> optimal makespan. Not tasks.
JSSP_CLOSED = {"ta18": 1396, "ta22": 1600, "ta23": 1557, "ta25": 1595, "ta26": 1643, "ta27": 1680, "ta29": 1625, "ta33": 1791, "ta34": 1828, "ta45": 1997, "dmu01": 2563, "dmu02": 2706, "dmu04": 2669, "dmu06": 3244, "dmu07": 3046, "dmu08": 3188, "dmu09": 3092, "dmu10": 2984, "dmu43": 3441, "swv06": 1667, "swv09": 1655, "swv12": 2972, "yn2": 904, "yn3": 892, "yn4": 967}
assert len(JSSP_CLOSED) == 25 and not set(JSSP) & set(JSSP_CLOSED)

# ---- Sorting networks (Dobbelaere's list): n -> (best size, depth of that network, proven size lower bound).
# ---- n = 16, 17, 18 already exist as suite tasks.
SORTNET = {19: (85, 12, 73), 20: (91, 12, 78), 21: (99, 15, 84), 22: (106, 13, 89), 23: (114, 14, 95), 24: (120, 13, 100),
           25: (130, 15, 106), 26: (138, 15, 111), 27: (147, 16, 117), 28: (155, 14, 123), 29: (164, 15, 129),
           30: (172, 14, 135), 31: (180, 14, 141), 32: (185, 14, 147)}

# ---- Snake-in-the-box, dimensions 10..13 (9 already exists). The July 2026 census (arXiv:2607.15270) improved all of
# ---- them; only the dimension-9 value (191, previously 190) was read, so no record value is stored for these.
SNAKE = [10, 11, 12, 13]

# ---- EPFL combinational benchmarks, LUT-6 records (lsils/benchmarks best_results, 2026).
# name -> (folder, file stem, best LUT count, best levels)
EPFL = {
    "adder": ("arithmetic", "adder", 129, 5), "barrel-shifter": ("arithmetic", "bar", 512, 4),
    "divisor": ("arithmetic", "div", 3082, 170), "hypotenuse": ("arithmetic", "hyp", 36451, 458),
    "log2": ("arithmetic", "log2", 5986, 51), "max": ("arithmetic", "max", 511, 6),
    "multiplier": ("arithmetic", "multiplier", 4282, 25), "sine": ("arithmetic", "sin", 994, 10),
    "square-root": ("arithmetic", "sqrt", 2958, 150), "square": ("arithmetic", "square", 2894, 10),
    "round-robin-arbiter": ("random_control", "arbiter", 259, 5), "alu-control-unit": ("random_control", "ctrl", 25, 2),
    "coding-cavlc": ("random_control", "cavlc", 46, 3), "decoder": ("random_control", "dec", 264, 2),
    "i2c-controller": ("random_control", "i2c", 173, 3), "int-to-float": ("random_control", "int2float", 17, 2),
    "memory-controller": ("random_control", "mem_ctrl", 1669, 5), "priority-encoder": ("random_control", "priority", 86, 4),
    "lookahead-router": ("random_control", "router", 18, 3), "voter": ("random_control", "voter", 1145, 11),
}

# ---- TDC ADMET group: TDC name -> (metric, top leaderboard score read on 2026-11-02 or None, leader)
ADMET = {
    "Caco2_Wang": ("mae", 0.256, "CaliciBoost"), "HIA_Hou": ("auroc", 0.993, "MiniMol"),
    "Pgp_Broccatelli": ("auroc", 0.938, "MapLight + GNN"), "Bioavailability_Ma": ("auroc", 0.942, "MiniMol"),
    "Lipophilicity_AstraZeneca": ("mae", 0.456, "MiniMol"), "Solubility_AqSolDB": ("mae", 0.741, "MiniMol"),
    "BBB_Martins": ("auroc", 0.924, "MiniMol"), "PPBR_AZ": ("mae", 7.440, "Gradient Boost"),
    "VDss_Lombardo": ("spearman", 0.713, "MapLight + GNN"), "CYP2C9_Veith": ("auprc", 0.859, "MapLight + GNN"),
    "CYP2D6_Veith": ("auprc", 0.790, "MapLight + GNN"), "CYP3A4_Veith": ("auprc", 0.916, "MapLight + GNN"),
    "CYP2C9_Substrate_CarbonMangels": ("auprc", 0.474, "MiniMol"), "CYP2D6_Substrate_CarbonMangels": ("auprc", 0.736, "ContextPred"),
    "CYP3A4_Substrate_CarbonMangels": ("auroc", 0.667, "CFA"), "Half_Life_Obach": ("spearman", 0.576, "CFA"),
    "Clearance_Hepatocyte_AZ": ("spearman", 0.536, "CFA"), "Clearance_Microsome_AZ": ("spearman", 0.630, "MapLight + GNN"),
    "LD50_Zhu": ("mae", 0.552, "BaseBoosting"), "hERG": ("auroc", 0.880, "MapLight + GNN"), "AMES": ("auroc", 0.871, "ZairaChem"), "DILI": ("auroc", 0.956, "MiniMol"),
}
assert len(ADMET) == 22

# ---- G-set (Helmberg and Rendl; files at web.stanford.edu/~yyye/yyye/Gset/). Best-known cuts were not fetched except
# ---- the two 2025 reports (G81 14060, G63 27047); the rest stay None until read from a primary table.
GSET = [f"G{i}" for i in range(1, 68)] + ["G70", "G72", "G77", "G81"]
GSET_REPORTED = {"G81": 14060, "G63": 27047}
assert len(GSET) == 71
