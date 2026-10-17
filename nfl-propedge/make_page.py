"""Inject data.json into template.html.

Writes two files:
  index.html       page content for publishing as an artifact (no document wrapper)
  site/index.html  full standalone page for GitHub Pages (or the path given with --site)
"""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="?", default="data.json")
ap.add_argument("--site", default="site/index.html", help="Where to write the standalone page")
args = ap.parse_args()
data = open(args.data).read()
json.loads(data)
safe = data.replace("</", "<\\/")
body = open("template.html").read().replace("/*DATA*/", safe)
open("index.html", "w").write(body)
os.makedirs(os.path.dirname(os.path.abspath(args.site)), exist_ok=True)
page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '</head><body>' + body + '</body></html>')
open(args.site, "w").write(page)
print("index.html", len(body), "|", args.site, len(page))
