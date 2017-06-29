
import numpy as np
import pandas as pd

action_labels = {0: "Insert",
                 1: "Cancel",
                 2: "Modif/Insert",
                 3: "Modif/Cancel",
                 4: "Modif",
                 5: "Flicker"}

aggressive_labels = {0: "Passive, deep in the book",
                     1: "Passive, in the book",
                     2: "Passive, at the best limit",
                     3: "Passive, inside the spread",
                     4: "Aggressive",
                     5: "Very aggressive"}


def SetDistances(df, ticksize, symmetric=True):
    if symmetric:
        df["bid0_dist"] = ((df["price"] - df["bid0"]) / ticksize).round().fillna(0).astype("<i8")
        df["ask0_dist"] = ((df["price"] - df["ask0"]) / ticksize).round().fillna(0).astype("<i8")
    else:
        prc_ref = df["bid0"].where(df["side"] == 1, df["ask0"])
        df["best_dist"] = ((df["price"] - prc_ref) / ticksize).round().fillna(0).astype("<i8") * df["side"]


def SetCrossedDistances(df, ticksize):
    prc0 = df["bid0"].where(df["side"] == 1, df["ask0"])
    prc1 = df["ask0"].where(df["side"] == 1, df["bid0"])
    trd = df["traded_max_price"].where(df["side"] == 1, df["traded_min_price"])
    
    dist0 = ((df["price"] - prc0) * df["side"] / ticksize).round().fillna(1).astype("<i8") - 1
    dist1 = (((trd        - prc1) * df["side"] / ticksize).round().fillna(-1).astype("<i8") + 1).clip(0)
    df["cross_dist"] = dist0.where(dist0 < 0, dist1)
    
    best_qty = df["bid0_quantity"].where(df["side"] == 1, df["ask0_quantity"])
    df.loc[df["action"].isin([1, 3]) & (df["cross_dist"] == -1) & (df["quantity"] >= best_qty), "cross_dist"] = 0
    
    best_qty = df["ask0_quantity"].where(df["side"] == 1, df["bid0_quantity"])
    df.loc[df["action"].isin([0, 2, 5, 6]) & (df["cross_dist"] == 1) & (df["quantity"] >= best_qty), "cross_dist"] = 2


def SetBestDistances(df, ticksize, fill_value=-999):
    prc0 = df["bid0"].where(df["side"] == 1, df["ask0"])
    prc1 = df["ask0"].where(df["side"] == 1, df["bid0"])
    
    trd = df["traded_max_price"].where(df["side"] == 1, df["traded_min_price"])
    prc = df["price"].where(df["traded_quantity"] == 0, trd)
    df["pass_dist"] = ((prc - prc0) * df["side"] / ticksize).round().fillna(fill_value).astype("<i8")
    df["aggr_dist"] = ((prc - prc1) * df["side"] / ticksize).round().fillna(fill_value).astype("<i8")

    qty0 = df["bid0_quantity"].where(df["side"] == 1, df["ask0_quantity"])
    qty1 = df["ask0_quantity"].where(df["side"] == 1, df["bid0_quantity"])
    
    df.loc[(df["aggr_dist"] == 0) & (df["quantity"] >= qty1), "aggr_dist"] = -1
    df.loc[(df["pass_dist"] == 0) & (df["quantity"] >= qty0) & df["action"].isin([1, 3]), "pass_dist"] = 1


def SetAggressivity2(df, cutoff):
    df["aggressivity"] = (df["pass_dist"] >= -cutoff).astype("<i8")
    df.loc[df["pass_dist"] == 0, "aggressivity"] = 2
    df.loc[(df["pass_dist"] > 0) & (df["aggr_dist"] < 0), "aggressivity"] = 3
    df.loc[df["aggr_dist"] == 0, "aggressivity"] = 4
    df.loc[df["aggr_dist"] > 0, "aggressivity"] = 5


def SetAggressivity(df, cutoff):
    df["aggressive"] = df["action"]
    
    df.loc[(df["side"] == 1) & (df["ask0_dist"] >= 0) & (df["quantity"] >= df["ask0_quantity"]), "aggressive"] = 5
    df.loc[(df["side"] == 1) & (df["ask0_dist"] >= 0) & (df["quantity"] < df["ask0_quantity"]), "aggressive"] = 4
    df.loc[(df["side"] == 1) & (df["ask0_dist"] <  0) & (df["bid0_dist"] > 0), "aggressive"] = 3
    df.loc[(df["side"] == 1) & (df["bid0_dist"] == 0), "aggressive"] = 2
    df.loc[(df["side"] == 1) & (df["bid0_dist"] <  0) & (df["bid0_dist"] >= -cutoff), "aggressive"] = 1
    df.loc[(df["side"] == 1) & (df["bid0_dist"] < -cutoff), "aggressive"] = 0
    
    df.loc[(df["side"] == 1) & (df["bid0_dist"] == 0) & ((df["action"] == 1) | (df["action"] == 3)) & (df["quantity"] == df["bid0_quantity"]), "aggressive"] = 3
    
    df.loc[(df["side"] == -1) & (df["bid0_dist"] <= 0) & (df["quantity"] >= df["bid0_quantity"]), "aggressive"] = 5
    df.loc[(df["side"] == -1) & (df["bid0_dist"] <= 0) & (df["quantity"] < df["bid0_quantity"]), "aggressive"] = 4
    df.loc[(df["side"] == -1) & (df["bid0_dist"] >  0) & (df["ask0_dist"] < 0), "aggressive"] = 3
    df.loc[(df["side"] == -1) & (df["ask0_dist"] == 0), "aggressive"] = 2
    df.loc[(df["side"] == -1) & (df["ask0_dist"] >  0) & (df["ask0_dist"] <= cutoff), "aggressive"] = 1
    df.loc[(df["side"] == -1) & (df["ask0_dist"] >  cutoff), "aggressive"] = 0
    
    df.loc[(df["side"] == -1) & (df["ask0_dist"] == 0) & ((df["action"] == 1) | (df["action"] == 3)) & (df["quantity"] == df["ask0_quantity"]), "aggressive"] = 3


def SetMid(df):
    df["mid"] = .5 * (df["bid0"] + df["ask0"])


def SetHalfSpread(df):
    df["hsp"] = .5 * (df["ask0"] - df["bid0"])


def SetImbalance0(df):
    df["imb0"] = df["bid0_quantity"] - df["ask0_quantity"]
    df["imb0_sign"] = np.sign(df["imb0"])


def SetImbalance1(df):
    df["imb1"] = df["imb0"].copy()
    df["imb1"] += df["bid1_quantity"] - df["ask1_quantity"]
    df["imb1_sign"] = np.sign(df["imb1"])


def SetImbalance2(df):
    df["imb2"] = df["imb1"].copy()
    df["imb2"] += df["bid2_quantity"] - df["ask2_quantity"]
    df["imb2_sign"] = np.sign(df["imb2"])


def SetNormedImbalance0(df):
    df["nmb0"] = df["imb0"] / (df["bid0_quantity"] + df["ask0_quantity"])


def SetMaxImbalance0(df):
    df["mmb0"] = df["imb0"] / df[["bid0_quantity", "ask0_quantity"]].max(axis=1)


def SetMmp0(df):
    df["mmp0"] = df["nmb0"] * df["hsp"] + df["mid"]


#def SetEmp0(df):
#    df["emp0"] = df["mid"].copy()
#    a = df.loc[df["imb0"] > 0, ["mmb0", "ask0", "ask1"]]
#    df.loc[df["imb0"] > 0, "emp0"] += .5 * a["mmb0"] * (a["ask1"] - a["ask0"])
#    
#    b = df.loc[df["imb0"] < 0, ["mmb0", "bid0", "bid1"]]
#    df.loc[df["imb0"] < 0, "emp0"] += .5 * b["mmb0"] * (b["bid0"] - b["bid1"])

def SetEmp0(df):
    df["emp0"]  = df["bid0"] * df["bid0_quantity"]
    df["emp0"] += df["ask0"] * df["ask0_quantity"]
    
    a = df.loc[df["imb0"] > 0, ["imb0", "ask1", "ask1_quantity", "ask2"]]
    aq = a[["imb0", "ask1_quantity"]].min(axis=1)
    df.loc[df["imb0"] > 0, "emp0"] += a["ask1"] * aq
    df.loc[df["imb0"] > 0, "emp0"] += a["ask2"] * (a["imb0"] - aq)
    
    b = df.loc[df["imb0"] < 0, ["imb0", "bid1", "bid1_quantity", "bid2"]]
    bq = (-b["imb0"]).clip(upper=b["bid1_quantity"])
    df.loc[df["imb0"] < 0, "emp0"] += b["bid1"] * bq
    df.loc[df["imb0"] < 0, "emp0"] -= b["bid2"] * (b["imb0"] + bq)
    
    v = df[["bid0_quantity", "ask0_quantity"]].max(axis=1)
    df["emp0"] /= 2 * v
    

def SetEmp1(df):
    df["emp1"]  = df["bid0"] * df["bid0_quantity"]
    df["emp1"] += df["bid1"] * df["bid1_quantity"]
    df["emp1"] += df["ask0"] * df["ask0_quantity"]
    df["emp1"] += df["ask1"] * df["ask1_quantity"]
    
    a = df.loc[df["imb1"] > 0, ["imb1", "ask2", "ask2_quantity", "ask3"]]
    aq = a[["imb1", "ask2_quantity"]].min(axis=1)
    df.loc[df["imb1"] > 0, "emp1"] += a["ask2"] * aq
    df.loc[df["imb1"] > 0, "emp1"] += a["ask3"] * (a["imb1"] - aq)
    
    b = df.loc[df["imb1"] < 0, ["imb1", "bid2", "bid2_quantity", "bid3"]]
    bq = (-b["imb1"]).clip(upper=b["bid2_quantity"])
    df.loc[df["imb1"] < 0, "emp1"] += b["bid2"] * bq
    df.loc[df["imb1"] < 0, "emp1"] -= b["bid3"] * (b["imb1"] + bq)
    
    bq = df["bid0_quantity"] + df["bid1_quantity"]
    aq = df["ask0_quantity"] + df["ask1_quantity"]
    v = pd.concat([bq, aq], axis=1)
    df["emp1"] /= 2 * v.max(axis=1)


def SetEmp2(df):
    df["emp2"]  = df["bid0"] * df["bid0_quantity"]
    df["emp2"] += df["bid1"] * df["bid1_quantity"]
    df["emp2"] += df["bid2"] * df["bid2_quantity"]
    df["emp2"] += df["ask0"] * df["ask0_quantity"]
    df["emp2"] += df["ask1"] * df["ask1_quantity"]
    df["emp2"] += df["ask2"] * df["ask2_quantity"]
    
    a = df.loc[df["imb2"] > 0, ["imb2", "ask3", "ask3_quantity", "ask4"]]
    aq = a[["imb2", "ask3_quantity"]].min(axis=1)
    df.loc[df["imb2"] > 0, "emp2"] += a["ask3"] * aq
    df.loc[df["imb2"] > 0, "emp2"] += a["ask4"] * (a["imb2"] - aq)
    
    b = df.loc[df["imb2"] < 0, ["imb2", "bid3", "bid3_quantity", "bid4"]]
    bq = (-b["imb2"]).clip(upper=b["bid3_quantity"])
    df.loc[df["imb2"] < 0, "emp2"] += b["bid3"] * bq
    df.loc[df["imb2"] < 0, "emp2"] -= b["bid4"] * (b["imb2"] + bq)
    
    bq = df["bid0_quantity"] + df["bid1_quantity"] + df["bid2_quantity"]
    aq = df["ask0_quantity"] + df["ask1_quantity"] + df["ask2_quantity"]
    v = pd.concat([bq, aq], axis=1)
    df["emp2"] /= 2 * v.max(axis=1)


def SetFairPrice(df, factor=.5):
    b_v_b = df["bid0"] * df["bid0_quantity"]
    a_v_a = df["ask0"] * df["ask0_quantity"]
    
    b_v_a = df["bid0"] * df["ask0_quantity"]
    a_v_b = df["ask0"] * df["bid0_quantity"]
    
    p1 = (1 + factor) * (b_v_a + a_v_b)
    p2 = (1 - factor) * (b_v_b + a_v_a)
    
    v_a_b = df["ask0_quantity"] + df["bid0_quantity"]
    
    df["fair_price"] = (p1 + p2) / (2 * v_a_b)
