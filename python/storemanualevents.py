
import datetime as dt
import numpy as np
import pandas as pd
import hashlib
import tables


actions = {"E": 0, "C": 1, "ME": 2, "MC": 3, "M": 4, "L": 5, "S": 6}


def DigestTrader(trader):
    trader_hasher = hashlib.blake2b(digest_size=6)
    trader_hasher.update(trader.encode())
    return int(trader_hasher.hexdigest(), 16)


DigestTraders = np.vectorize(DigestTrader)


class Bars(tables.IsDescription):
    session_date = tables.Int64Col()
    first_time = tables.Int64Col()
    last_time = tables.Int64Col()
    first_price = tables.Float64Col()
    last_price = tables.Float64Col()
    high_price = tables.Float64Col()
    low_price = tables.Float64Col()
    
    volume = tables.Int64Col()
    trade_nb = tables.Int64Col()


class Order(tables.IsDescription):
    time = tables.Int64Col()
    end_time = tables.Int64Col()
    order_id = tables.Int64Col()
    trader = tables.Int64Col()
    action = tables.Int8Col()
    side = tables.Int8Col()
    crossed = tables.BoolCol()
    halted = tables.Int8Col()
    price = tables.Float64Col()
    quantity = tables.Int64Col()
    visible = tables.Int64Col()
    bid0 = tables.Float64Col()
    bid1 = tables.Float64Col()
    bid2 = tables.Float64Col()
    bid3 = tables.Float64Col()
    bid4 = tables.Float64Col()
    ask0 = tables.Float64Col()
    ask1 = tables.Float64Col()
    ask2 = tables.Float64Col()
    ask3 = tables.Float64Col()
    ask4 = tables.Float64Col()
    bid0_quantity = tables.Int64Col()
    bid1_quantity = tables.Int64Col()
    bid2_quantity = tables.Int64Col()
    bid3_quantity = tables.Int64Col()
    bid4_quantity = tables.Int64Col()
    ask0_quantity = tables.Int64Col()
    ask1_quantity = tables.Int64Col()
    ask2_quantity = tables.Int64Col()
    ask3_quantity = tables.Int64Col()
    ask4_quantity = tables.Int64Col()
    bid0_visible = tables.Int64Col()
    bid1_visible = tables.Int64Col()
    bid2_visible = tables.Int64Col()
    bid3_visible = tables.Int64Col()
    bid4_visible = tables.Int64Col()
    ask0_visible = tables.Int64Col()
    ask1_visible = tables.Int64Col()
    ask2_visible = tables.Int64Col()
    ask3_visible = tables.Int64Col()
    ask4_visible = tables.Int64Col()


class Order2(Order):
    traded_price = tables.Float64Col()
    traded_max_price = tables.Float64Col()
    traded_min_price = tables.Float64Col()
    traded_weighted_price = tables.Float64Col()
    traded_quantity = tables.Int64Col()
    aggr_quantity = tables.Int64Col()
    trades = tables.Int64Col()
    

def ReadStore(h5file, table_name, index="time", convert_time_index=True):
    node = h5file.get_node(table_name)
    df = pd.DataFrame.from_records(node.read(), index=index)
    if convert_time_index and index is not None:
        return df.set_index(df.index.map(pd.Timestamp))
    else:
        return df

def ReadTable(node, index="time", convert_time_index=True, time_cols=["begin_time", "end_time"]):
    df = pd.DataFrame.from_records(node.read(), index=index)
    if index is not None:
        if convert_time_index:
            df.set_index(df.index.map(pd.Timestamp), inplace=True)
        df.index.name = index
    
    for i in time_cols:
        if i in df:
            df[i]= df[i].apply(pd.Timestamp)
    return df


def StoreManualEvents(snap, h5file, table_name, end_time=dt.time(23, 59)):
    
    traders = {}
    for i in snap.evt["trader"].unique():
        #traders[i] = int(i[:8])
        traders[i] = DigestTrader(i)
    
    if snap.symbol in h5file.root:
        group = h5file.get_node("/" + snap.symbol)
    else:
        group = h5file.create_group("/", snap.symbol)
    table = h5file.create_table(group, table_name, Order)
    order = table.row
    
    offset = 0
    is_crossed = True
    last_oid = None
    last_prc = np.nan
    last_qty = np.nan
    last_vis = np.nan
    last_bids = None
    last_asks = None
    last_bids_qty = None
    last_asks_qty = None
    last_bids_vis = None
    last_asks_vis = None
    #last_bid_nbr = np.nan
    #last_ask_nbr = np.nan
    
    levels = 5
    
    end_times = snap.evt.groupby("orderid")["time"].last()
    
    # loop on all events
    for k, (evt, booklet) in enumerate(snap.generateBooks()):
        
        # dont consider events after end_time
        if evt["time"].time() >= end_time:
            break

        typ = evt["event"]
        
        keep = False
        if typ == "E" or typ == "C":
            keep = True
        elif typ == "M":
            if evt["isStart"]:
                keep = True
        elif typ == "L" or typ == "S":
            keep = True
        
        if keep:
            prc = evt["price"]
            qty = evt["quantity"]
            vis = evt["visible"]
            oid = evt["orderid"]

            # buy order
            if evt["dir"]:                    
                side = 1
                
                # split modifications in two types:
                # increase aggressivity -> insert
                # decrease aggressivity -> cancel
                if typ == "M" and oid == last_oid:
                    if last_prc > prc:
                        prc = last_prc
                        qty = last_qty
                        vis = last_vis
                        typ = "MC"
                    elif last_prc < prc:
                        typ = "ME"
                    elif last_qty > qty:
                        qty = last_qty - qty
                        typ = "MC"
                    elif last_qty < qty:
                        qty = qty - last_qty
                        typ = "ME"
                    
            # sell order
            else:                    
                side = -1
                
                # split modifications in two types:
                # increase aggressivity -> insert
                # decrease aggressivity -> cancel
                if typ == "M" and oid == last_oid:
                    if last_prc < prc:
                        prc = last_prc
                        qty = last_qty
                        vis = last_vis
                        typ = "MC"
                    elif last_prc > prc:
                        typ = "ME"
                    elif last_qty > qty:
                        qty = last_qty - qty
                        typ = "MC"
                    elif last_qty < qty:
                        qty = qty - last_qty
                        typ = "ME"

            # record the event
            order["time"] = evt["time"].to_datetime64().astype("<i8") + offset
            order["end_time"] = end_times[oid].astype("<i8")
            order["order_id"] = oid
            order["trader"] = traders[evt["trader"]]
            order["action"] = actions[typ]
            order["side"] = side
            order["crossed"] = is_crossed
            order["halted"] = evt["halt"]
            order["price"] = prc
            order["quantity"] = qty
            order["visible"] = vis
            
            if last_bids is not None:
                for i in range(levels):
                    bid_key = "bid" + str(i)
                    order[bid_key] = last_bids[i]
                    order[bid_key + "_quantity"] = last_bids_qty[i]
                    order[bid_key + "_visible"] = last_bids_vis[i]
                    
                    ask_key = "ask" + str(i)
                    order[ask_key] = last_asks[i]
                    order[ask_key + "_quantity"] = last_asks_qty[i]
                    order[ask_key + "_visible"] = last_asks_vis[i]
            else:
                for i in range(levels):
                    bid_key = "bid" + str(i)
                    order[bid_key] = np.nan
                    order[bid_key + "_quantity"] = 0
                    order[bid_key + "_visible"] = 0
                    
                    ask_key = "ask" + str(i)
                    order[ask_key] = np.nan
                    order[ask_key + "_quantity"] = 0
                    order[ask_key + "_visible"] = 0

            order.append()
            
            is_crossed = True
            offset += 1

        # record the bid, ask and mid for future use
        if evt["lastInGroup"]:
            offset = 0
            if evt["halt"] == -1 and not booklet.crossed():
                last_bids = booklet.bid_prices(levels)
                last_asks = booklet.ask_prices(levels)
                
                last_bids_qty = booklet.get_volumes("B", last_bids)
                last_asks_qty = booklet.get_volumes("S", last_asks)
                last_bids_vis = booklet.get_volumes("BV", last_bids)
                last_asks_vis = booklet.get_volumes("SV", last_asks)
                
                is_crossed = False

        # remember order caracteristics when it is a modif
        if ~keep and typ == "M":
            last_oid = evt["orderid"]
            last_prc = evt["price"]
            last_qty = evt["quantity"]
            last_vis = evt["visible"]

    table.flush()


def StoreTrades(orders_file, trades_file, out_file, group_name, table_name):
    node = orders_file.get_node("/" + group_name + "/" + table_name)
    orders = ReadTable(node)
    
    node = trades_file.get_node("/" + group_name + "/" + table_name)
    trades = ReadTable(node)
    
    trades["weighted_price"] = trades["price"] * trades["quantity"]
    trades["count"] = 1
    
    orders["traded_price"] = 0.
    orders["traded_weighted_price"] = 0.
    orders["traded_max_price"] = 0.
    orders["traded_min_price"] = 0.
    orders["traded_quantity"] = 0
    orders["aggr_quantity"] = 0
    orders["trades"] = 0
    
    i1 = orders.reset_index().set_index(["order_id", "time"]).sort_index().index
    
    for key in ["buy_order_id", "sell_order_id"]:
        t = trades.reset_index().set_index([key, "time"]).sort_index()
        a = np.searchsorted(i1, t.index, "left") - 1
        
        good = t.index.get_level_values(key) == i1[a].get_level_values("order_id")
        a = a[good]
        t = t[good]
        
        g = t.groupby(i1[a].get_level_values("time"))["price", "quantity", "weighted_price", "count"]
        b_sum = g.sum()
        b_max = g.max()
        b_min = g.min()
        
        orders["traded_price"] = orders["traded_price"].add(b_sum["price"] / b_sum["count"], fill_value=0.)
        orders["traded_weighted_price"] = orders["traded_weighted_price"].add(b_sum["weighted_price"] / b_sum["quantity"], fill_value=0.)
        orders["traded_max_price"] = orders["traded_max_price"].add(b_max["price"], fill_value=0.)
        orders["traded_min_price"] = orders["traded_min_price"].add(b_min["price"], fill_value=0.)
        orders["traded_quantity"] = orders["traded_quantity"].add(b_sum["quantity"], fill_value=0).astype("<i8")
        orders["trades"] = orders["trades"].add(b_sum["count"], fill_value=0).astype("<i8")
        
        side = 1 if key == "buy_order_id" else -1
        t = trades[trades["side"] == side].reset_index().set_index([key, "time"]).sort_index()
        a = np.searchsorted(i1, t.index, "left") - 1
        
        good = t.index.get_level_values(key) == i1[a].get_level_values("order_id")
        a = a[good]
        t = t[good]
        
        g = t.groupby(i1[a].get_level_values("time"))["price", "quantity", "weighted_price", "count"]
        b_sum = g.sum()
        
        orders["aggr_quantity"] = orders["aggr_quantity"].add(b_sum["quantity"], fill_value=0).astype("<i8")
        
    orders.reset_index(inplace=True)
    orders["time"] = orders["time"].values.astype("<i8")
    
    if group_name in out_file.root:
        group = out_file.get_node("/" + group_name)
    else:
        group = out_file.create_group("/", group_name)
    table = out_file.create_table(group, table_name, Order2)
    table.append(orders.sort_index(axis=1).to_records(index=False))
    table.flush()

    
def MergeEvents(df):
    a = df.shift(-1).fillna(0)
    
    same_trader = (df["trader"] == a["trader"]) & (df["side"] == a["side"])
    insert_cancel = (df["action"] == 0) & (a["action"] == 1)
    cancel_insert = (df["action"] == 1) & (a["action"] == 0)
    same_price = df["price"] == a["price"]
    same_qty = df["quantity"] == a["quantity"]
    no_trade = df["traded_quantity"] == 0
    
    
    filt = same_trader & cancel_insert & same_price & same_qty & no_trade
    df.loc[filt, "action"] = 4
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt = filt
    
    filt = same_trader & insert_cancel & same_price & same_qty & no_trade
    df.loc[filt, "action"] = 5
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & cancel_insert & same_price & (df["quantity"] < a["quantity"])
    df.loc[filt, "action"] = 2
    df.loc[filt, ["quantity", "visible"]] = a.loc[filt , ["quantity", "visible"]].astype("<i8")
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & cancel_insert & same_price & no_trade & (df["quantity"] > a["quantity"])
    df.loc[filt, "action"] = 3
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & insert_cancel & same_price & no_trade & (df["quantity"] > a["quantity"])
    df.loc[filt, "action"] = 2
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & insert_cancel & same_price & no_trade & (df["quantity"] < a["quantity"])
    df.loc[filt, "action"] = 3
    df.loc[filt, ["quantity", "visible"]] = a.loc[filt , ["quantity", "visible"]].astype("<i8")
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & (df["side"] == 1) & cancel_insert & no_trade & (df["price"] < a["price"])
    df.loc[filt, "action"] = 2
    df.loc[filt, "price"] = a.loc[filt , "price"]
    df.loc[filt, ["quantity", "visible"]] = a.loc[filt , ["quantity", "visible"]].astype("<i8")
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & (df["side"] == 1) & cancel_insert & no_trade & (df["price"] > a["price"])
    df.loc[filt, "action"] = 3
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & (df["side"] == 1) & insert_cancel & no_trade & (df["price"] < a["price"])
    df.loc[filt, "action"] = 3
    df.loc[filt, "price"] = a.loc[filt , "price"]
    df.loc[filt, ["quantity", "visible"]] = a.loc[filt , ["quantity", "visible"]].astype("<i8")
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & (df["side"] == 1) & insert_cancel & no_trade & (df["price"] > a["price"])
    df.loc[filt, "action"] = 2
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    
    filt = same_trader & (df["side"] == -1) & cancel_insert & no_trade & (df["price"] > a["price"])
    df.loc[filt, "action"] = 2
    df.loc[filt, "price"] = a.loc[filt , "price"]
    df.loc[filt, ["quantity", "visible"]] = a.loc[filt , ["quantity", "visible"]].astype("<i8")
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & (df["side"] == -1) & cancel_insert & no_trade & (df["price"] < a["price"])
    df.loc[filt , "action"] = 3
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    filt = same_trader & (df["side"] == -1) & insert_cancel & no_trade & (df["price"] > a["price"])
    df.loc[filt, "action"] = 3
    df.loc[filt, "price"] = a.loc[filt , "price"]
    df.loc[filt, ["quantity", "visible"]] = a.loc[filt , ["quantity", "visible"]].astype("<i8")
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    
    filt = same_trader & (df["side"] == -1) & insert_cancel & no_trade & (df["price"] < a["price"])
    df.loc[filt, "action"] = 2
    same_trader ^= filt
    filt = filt.shift(1).fillna(False)
    same_trader ^= filt
    not_filt |= filt
    
    df = df[~not_filt]


class Trade(tables.IsDescription):
    time = tables.Int64Col()
    buy_order_id = tables.Int64Col()
    sell_order_id = tables.Int64Col()
    buyer = tables.Int64Col()
    seller = tables.Int64Col()
    price = tables.Float64Col()
    quantity = tables.Int64Col()    
    side = tables.Int8Col()


def StoreAllTrades(snap, h5file, table_name):
    traders = {}
    for i in np.unique(np.hstack([snap.tra["buyer"].unique(), snap.tra["seller"].unique()])):
        traders[i] = DigestTrader(i)
    
    if snap.symbol in h5file.root:
        group = h5file.get_node("/" + snap.symbol)
    else:
        group = h5file.create_group("/", snap.symbol)
    table = h5file.create_table(group, table_name, Trade)
    trade = table.row

    for row in snap.tra.itertuples():
        trade["time"] = row.time.to_datetime64().astype("<i8")
        trade["buy_order_id"] = row.buyOrderid
        trade["sell_order_id"] = row.sellOrderid
        trade["buyer"] = traders[row.buyer]
        trade["seller"] = traders[row.seller]
        trade["price"] = row.price
        trade["quantity"] = row.quantity
        if row.param == "A":
            trade["side"] = 1
        elif row.param == "V":
            trade["side"] = -1
        else:
            trade["side"] = 0
        trade.append()
    table.flush()


class Halt(tables.IsDescription):
    begin_time = tables.Int64Col()
    end_time = tables.Int64Col()


def StoreHalts(snap, h5file, table_name):
    if snap.symbol in h5file.root:
        group = h5file.get_node("/" + snap.symbol)
    else:
        group = h5file.create_group("/", snap.symbol)
    table = h5file.create_table(group, table_name, Halt)
    halt = table.row

    for row in snap.halt.itertuples():
        halt["begin_time"] = row.start.to_datetime64().astype("<i8")
        halt["end_time"] = row.end.to_datetime64().astype("<i8")
    
        halt.append()
    table.flush()
