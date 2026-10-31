"""Check published dataset project pages. These are not assumed to be archive URLs."""
import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse
import requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",default="datasets/sources.csv")
    a=p.parse_args()
    with Path(a.catalog).open(newline="",encoding="utf8") as f:
        for row in csv.DictReader(f):
            url=row["source_or_project_url"]
            try:
                if urlparse(url).scheme != "https":
                    raise ValueError("Only HTTPS source pages are permitted")
                # Do not follow an external redirect to another URL scheme.
                # A redirect is recorded as a source-page response, not an archive.
                with requests.head(url,allow_redirects=False,timeout=20,
                        headers={"User-Agent":"AIRFormer-reproduction/1.0"}) as res:
                    res.raise_for_status()
                    print(f"REACHABLE {res.status_code} {row['dataset']} {url}")
            except Exception as e:
                print(f"UNVERIFIED {row['dataset']} {url}: {type(e).__name__}: {e}")
    print("This command checks project pages only; download archives from owner instructions after verifying format, terms, and pairing.")


if __name__=="__main__": main()
