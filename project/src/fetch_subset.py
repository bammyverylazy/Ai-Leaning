import argparse
from pathlib import PurePosixPath

import pandas as pd
from remotezip import RemoteZip

import config as C

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
parser = argparse.ArgumentParser(description="Download ASL Citizen metadata or selected videos.")
parser.add_argument(
    "step",
    nargs="?",
    choices=["list", "csv", "videos"],
    default="csv",
    help="list archive contents, download metadata CSVs, or download selected videos",
)
step = parser.parse_args().step

with RemoteZip(URL) as z:
    names = z.namelist()
    if step == "list":
        print(len(names))
        print("\n".join(names[:15]))
    elif step == "csv":
        for n in names:
            if n.lower().endswith(".csv"):
                z.extract(n, C.RAW)
                print("got", n)
    else:
        df = pd.read_csv(C.DATA / "subset_all.csv")
        wanted = set(df.video)
        matched = [n for n in names if PurePosixPath(n).name in wanted]
        matched_names = {PurePosixPath(n).name for n in matched}
        missing = sorted(wanted - matched_names)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected videos were not found in the archive: {missing[:10]}"
            )

        todo = [n for n in matched if not (C.RAW / n).exists()]
        print(f"{len(matched)} selected videos; {len(todo)} still need downloading")
        for i, n in enumerate(todo, 1):
            z.extract(n, C.RAW)
            if i % 50 == 0:
                print(i, "/", len(todo))
        print(f"Downloaded {len(todo)} videos")
