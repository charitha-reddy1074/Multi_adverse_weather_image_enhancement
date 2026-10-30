"""Check published dataset project pages. These are not assumed to be archive URLs."""
import argparse
import csv
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",default="datasets/sources.csv")
    a=p.parse_args()
    with Path(a.catalog).open(newline="",encoding="utf8") as f:
        for row in csv.DictReader(f):
            url=row["source_or_project_url"]
            try:
                req=Request(url,method="HEAD",headers={"User-Agent":"AIRFormer-reproduction/1.0"})
                with urlopen(req,timeout=20) as res:
                    print(f"REACHABLE {res.status} {row['dataset']} {url}")
            except Exception as e:
                print(f"UNVERIFIED {row['dataset']} {url}: {type(e).__name__}: {e}")
    print("This command checks project pages only; download archives from owner instructions after verifying format, terms, and pairing.")


if __name__=="__main__": main()
