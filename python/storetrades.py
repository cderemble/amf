
import numpy as np
import pandas as pd
import tables


class Trades(tables.IsDescription):
    time = tables.Int64Col()
    trader_id = tables.Int64Col()
    trade_id = tables.Int64Col()
    sequence_id = tables.Int64Col()
    side = tables.Int8Col()
    price = tables.Float64Col()
    quantity = tables.Int64Col()
    origin_id = tables.Int8Col()
    is_auction = tables.BoolCol()
    is_aggressor = tables.BoolCol()


def set_to_dict(in_set, out_dict):
    num = len(out_dict)
    for i in in_set:
        if not i in out_dict:
            out_dict[i] = num
            num += 1
    return out_dict


def store_trades(input_file_name, output_file_name):
    columns = ['Price',
               'Trader',
               'Isin',
               'Date',
               'TraderOrigin',
               'Quantity',
               'AuctionFlag',
               'Aggression',
               'Direction',
               'TradingClip']

    reader = pd.read_csv(input_file_name,
                         sep=";",
                         usecols=columns,
                         true_values=["OpeningAuction"],
                         false_values=["TradingPhase"],
                         parse_dates=[3],
                         chunksize=1000000)

    traders = set()
    origins = set()
    trader_ids = {}
    origin_ids = {}

    output_file = tables.open_file(output_file_name, mode="w")

    for chunk in reader:
        traders.update(chunk["Trader"].unique())
        origins.update(chunk["TraderOrigin"].unique())

        trader_ids = set_to_dict(traders, trader_ids)
        origin_ids = set_to_dict(origins, origin_ids)

        store_chunk(output_file, chunk, trader_ids, origin_ids)

    output_file.close()

    return trader_ids, origin_ids


def store_chunk(output_file,
                chunk,
                trader_ids,
                origin_ids):

    sides = {"B": 1, "S": -1}

    data = pd.DataFrame({"price": chunk["Price"]})
    data["quantity"] = chunk["Quantity"].astype("<i8")
    data["time"] = chunk["Date"]
    data["is_auction"] = chunk["AuctionFlag"]
    data["trader_id"] = chunk["Trader"].map(trader_ids).astype("<i8")
    data["origin_id"] = chunk["TraderOrigin"].map(origin_ids).astype("<i1")
    data["is_aggressor"] = (chunk["Direction"] == chunk["Aggression"]).fillna(False)
    data["side"] = chunk["Direction"].map(sides).astype("<i1")
    data["trade_id"] = chunk["TradingClip"].apply(lambda x: np.int64(x[4:]))
    data["sequence_id"] = chunk["TraderOrigin"].map(origin_ids).astype("<i8")

    session_dates = chunk["Date"].dt.date
    dates = session_dates.unique()
    isin_codes = chunk["Isin"].unique()

    for isin_code in isin_codes:
        if isin_code in output_file.root:
            group = output_file.get_node("/" + isin_code)
        else:
            group = output_file.create_group("/", isin_code)

        for date in dates:
            table_name = date.strftime("T%Y%m%d")

            filt = (chunk["Isin"] == isin_code) & (session_dates == date)

            if filt.any():
                if table_name in group:
                    table = output_file.get_node("/" + isin_code + "/" + table_name)
                else:
                    table = output_file.create_table(group, table_name, Trades)

                filtered_data = data[filt].sort_index(axis=0)
                table.append(filtered_data.sort_index(axis=1).to_records(index=False))
                table.flush()
