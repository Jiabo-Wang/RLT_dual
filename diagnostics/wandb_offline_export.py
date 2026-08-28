"""Read a wandb transaction log (``*.wandb``) offline and dump it to CSV.

The run in ``wandb/run-*/`` was produced on the training box, so there is no
server to query -- this walks the local leveldb-framed log instead.

    python diagnostics/wandb_offline_export.py \
        wandb/run-20260819_211931-biy4hbcy/run-biy4hbcy.wandb \
        outputs/pi05_vla_ft_report
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore


def _val(item):
    raw = item.value_json
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _key(item):
    if item.nested_key:
        return ".".join(item.nested_key)
    return item.key


def main(wandb_file: str, outdir: str) -> None:
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)

    history: list[dict] = []
    stats: list[dict] = []
    kinds = defaultdict(int)

    while True:
        try:
            data = ds.scan_data()
        except Exception as exc:  # truncated tail is normal for a killed run
            print(f"stopped scanning: {type(exc).__name__}: {exc}")
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        kind = rec.WhichOneof("record_type")
        kinds[kind] += 1

        if kind == "history":
            row = {_key(i): _val(i) for i in rec.history.item}
            history.append(row)
        elif kind == "stats":
            row = {i.key: _val(i) for i in rec.stats.item}
            row["_timestamp"] = rec.stats.timestamp.seconds + rec.stats.timestamp.nanos / 1e9
            stats.append(row)

    print("record kinds:", dict(kinds))
    print(f"history rows: {len(history)}  stats rows: {len(stats)}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for name, rows in (("history.csv", history), ("system.csv", stats)):
        if not rows:
            continue
        cols: list[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        # scalars only; drop list/dict-valued keys (e.g. loss_per_dim) into a side file
        scalar_cols = [c for c in cols if not any(isinstance(r.get(c), (list, dict)) for r in rows)]
        drop = [c for c in cols if c not in scalar_cols]
        if drop:
            print(f"{name}: non-scalar keys kept out of CSV -> {drop}")
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=scalar_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out / name}  ({len(rows)} rows, {len(scalar_cols)} cols)")
        print("  cols:", scalar_cols)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
