import numpy as np
import pandas as pd
import yaml
from tqdm import trange
import gzip


def labels():
    with open("data/labels.csv", "r") as file:
        df = pd.read_csv(file)
    return df


def config():
    with open("config.yml", "r") as file:
        contents = yaml.safe_load(file)
    return contents


def reference_table():
    ref_table_dir = config()["directories"]["ref_table"]
    ref_table = pd.read_csv(ref_table_dir)
    obj_id = np.array(ref_table["objid"])
    sample = np.array(ref_table["sample"])
    img_id = np.array(ref_table["asset_id"])
    return pd.DataFrame(np.array((sample, img_id)).T, columns=["SAMPLE", "IMG_ID"], index=obj_id)


def gz2_table5():
    with gzip.open(config()["directories"]["labels"], "rt", newline="\n") as file:
        content = file.readlines()
    headers = content[0].split(",")
    indexes = [i for i in range(len(headers)) if headers[i][-8:] == "debiased"]
    columns = [headers[i] for i in indexes]
    data = content[1:]
    object_id = []
    for idx in trange(len(data), desc="Acquiring Labels: "):
        data[idx] = data[idx][:-1]
        data[idx] = data[idx].split(",")
        object_id.append(np.int64(data[idx][2]))
        data[idx] = [data[idx][i] for i in indexes]
    df = pd.DataFrame(data=data, columns=columns, index=object_id)
    return df


def gz2_metadata_table():
    meta_table_dir = config()["directories"]["gal_zoo2_metadata_table"]
    metadata = pd.read_csv(meta_table_dir)
    object_id = metadata["OBJID"]
    eg = metadata["EXTINCTION_G"]
    er = metadata["EXTINCTION_R"]
    extinction = eg - er
    extinction.index = object_id
    return pd.DataFrame(extinction)
