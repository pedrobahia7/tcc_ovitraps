"""
Empirical Bayes Comparison Dashboard — Static HTML Export

Generates a self-contained HTML with all biweeks pre-computed.
GeoJSON + sector metadata embedded once; only z-values and case_counts
are repeated per biweek, keeping file size manageable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# =====================================================================
# PATHS & CONSTANTS
# =====================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PER_CAPITA_CSV = PROCESSED_DIR / "dengue_per_capita.csv"
GEOJSON_PATH = PROCESSED_DIR / "bh_sectors_2022_with_populations.geojson"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_HTML = RESULTS_DIR / "eb_comparison_dashboard.html"

MAP_CENTER_LAT = -19.9167
MAP_CENTER_LON = -43.9345
MAP_ZOOM = 11
MAPBOX_STYLE = "carto-positron"

RATE_COLORSCALE = [
    [0.0, "rgb(5,48,97)"],
    [0.15, "rgb(33,102,172)"],
    [0.3, "rgb(67,147,195)"],
    [0.45, "rgb(146,197,222)"],
    [0.5, "rgb(247,247,247)"],
    [0.6, "rgb(244,165,130)"],
    [0.75, "rgb(214,96,77)"],
    [0.9, "rgb(178,24,43)"],
    [1.0, "rgb(103,0,31)"],
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# =====================================================================
# DATA
# =====================================================================

log.info("Loading data …")
sectors_gdf = gpd.read_file(GEOJSON_PATH)
sectors_gdf["CD_SETOR"] = sectors_gdf["CD_SETOR"].astype(str)
GEOJSON_DATA = json.loads(sectors_gdf.to_json())

pc_df = pd.read_csv(PER_CAPITA_CSV)
pc_df["sector_id"] = pc_df["sector_id"].astype(str)
pc_df["biweek"] = pc_df["biweek"].astype(str)

ALL_BIWEEKS = sorted(pc_df["biweek"].unique())
log.info(
    "  %d rows | %d biweeks | %d sectors",
    len(pc_df),
    len(ALL_BIWEEKS),
    sectors_gdf.shape[0],
)

# =====================================================================
# CONSTANTS: ordered sector list & population (shared across biweeks)
# =====================================================================

SECTOR_IDS: list[str] = sectors_gdf["CD_SETOR"].tolist()
_SECTOR_IDX: dict[str, int] = {s: i for i, s in enumerate(SECTOR_IDS)}
N_SECTORS = len(SECTOR_IDS)

# Population per sector (from most recent biweek that has data for it)
_last_bw = ALL_BIWEEKS[-1]
_pop_series = (
    pc_df[pc_df["biweek"] == _last_bw]
    .set_index("sector_id")["population"]
    .reindex(SECTOR_IDS)
    .fillna(0)
    .astype(int)
)
SECTOR_POPULATIONS: list[int] = _pop_series.tolist()

# =====================================================================
# DATA EXTRACTION HELPERS
# =====================================================================


def _map_arrays(
    biweek_df: pd.DataFrame, col: str
) -> tuple[list[float], float, list[int]]:
    """Return (z_array, zmax, case_count_array) aligned to SECTOR_IDS."""
    sub = biweek_df.set_index("sector_id")[[col, "case_count"]]
    z_ser = sub[col].reindex(SECTOR_IDS).fillna(0.0)
    cc_ser = sub["case_count"].reindex(SECTOR_IDS).fillna(0).astype(int)
    zmax = float(max(z_ser.quantile(0.95), 1))
    return z_ser.tolist(), zmax, cc_ser.tolist()


def _scatter_payload(wdf: pd.DataFrame) -> dict:
    valid = wdf[
        (wdf["population"] > 0) & (wdf["cases_per_1000"] > 0)
    ].copy()
    if valid.empty:
        return {
            "sx": [], "sy": [], "sc": [], "st": [],
            "dxy": 0.0,
            "vx": [], "vy": [], "vt": [],
        }
    valid["log_pop"] = np.log10(valid["population"].clip(lower=1))
    valid["shrinkage"] = (
        valid["cases_per_1000"] - valid["eb_rate_per_1000"]
    ).abs()
    max_val = float(
        max(valid["cases_per_1000"].max(), valid["eb_rate_per_1000"].max())
    )
    return {
        "sx": valid["cases_per_1000"].tolist(),
        "sy": valid["eb_rate_per_1000"].tolist(),
        "sc": valid["log_pop"].tolist(),
        "st": [
            f"Sector: {s}<br>Pop: {p}<br>Crude: {c:.3f}<br>EB: {e:.3f}"
            for s, p, c, e in zip(
                valid["sector_id"],
                valid["population"],
                valid["cases_per_1000"],
                valid["eb_rate_per_1000"],
            )
        ],
        "dxy": max_val,
        "vx": valid["population"].tolist(),
        "vy": valid["shrinkage"].tolist(),
        "vt": [
            f"Sector: {s}<br>Pop: {p}<br>|Δ|: {d:.3f}"
            for s, p, d in zip(
                valid["sector_id"],
                valid["population"],
                valid["shrinkage"],
            )
        ],
    }


def _hist_payload(wdf: pd.DataFrame) -> dict:
    pos = wdf[wdf["cases_per_1000"] > 0]
    return {
        "crude": pos["cases_per_1000"].tolist() if not pos.empty else [],
        "eb": pos["eb_rate_per_1000"].tolist() if not pos.empty else [],
    }


# =====================================================================
# PRE-COMPUTE ALL BIWEEKS
# =====================================================================

log.info("Pre-computing %d biweeks …", len(ALL_BIWEEKS))
all_data: list[dict] = []
for bw in ALL_BIWEEKS:
    wdf = pc_df[pc_df["biweek"] == bw].copy()
    cz, czmax, cc = _map_arrays(wdf, "cases_per_1000")
    ez, ezmax, _ = _map_arrays(wdf, "eb_rate_per_1000")
    sp = _scatter_payload(wdf)
    hp = _hist_payload(wdf)
    all_data.append(
        {
            "bw": bw,
            "cz": cz, "czmax": czmax,
            "ez": ez, "ezmax": ezmax,
            "cc": cc,
            **sp,
            "hist": hp,
        }
    )
log.info("Pre-computation done.")

# =====================================================================
# HTML TEMPLATE  (__PLACEHOLDER__ avoids .format() curly-brace clash)
# =====================================================================

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>EB Comparison — Crude vs Smoothed</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body{font-family:Arial,sans-serif;padding:10px}
  h1{text-align:center;color:#2C3E50;margin-bottom:5px}
  .sub{text-align:center;color:#888;font-size:13px;margin-bottom:15px}
  #sw{margin:0 40px 20px 40px}
  #bwl{font-weight:bold;margin-bottom:4px}
  #bws{width:100%}
  .maps{display:flex}
  .maps>div{width:50%;height:500px}
  .charts{display:flex;margin:0 20px}
  .charts>div{width:33%;height:420px}
  hr{margin:20px 40px}
</style>
</head>
<body>
<h1>Empirical Bayes Comparison — Crude vs Smoothed Rates</h1>
<p class="sub">
  Marshall (1991) Poisson-Gamma EB smoothing.
  Small-population sectors shrunk toward global mean;
  large-population sectors nearly unchanged.
</p>
<div id="sw">
  <div id="bwl">Biweek: __FIRST_BW__</div>
  <input type="range" id="bws" min="0" max="__MAX_IDX__" value="0"/>
</div>
<div class="maps">
  <div id="map-crude"></div>
  <div id="map-eb"></div>
</div>
<hr/>
<div class="charts">
  <div id="scatter"></div>
  <div id="shrink"></div>
  <div id="hist"></div>
</div>
<script>
const GJ=__GEOJSON__;
const BW=__BIWEEKS__;
const AD=__ALL_DATA__;
const CS=__RATE_CS__;
const LOC=__LOCATIONS__;
const POP=__POPULATIONS__;
const ML={mapbox:{style:"carto-positron",center:{lat:__LAT__,lon:__LON__},zoom:__ZOOM__},margin:{l:0,r:0,t:35,b:0},uirevision:"constant"};

function hover(z,cc){
  return LOC.map((s,i)=>"Sector: "+s+"<br>Cases: "+cc[i]+"<br>Pop: "+POP[i]+"<br>Rate: "+z[i].toFixed(3));
}
function mT(z,zmax,cc,title){
  return {type:"choroplethmapbox",geojson:GJ,featureidkey:"properties.CD_SETOR",
    locations:LOC,z:z,zmin:0,zmax:zmax,colorscale:CS,
    marker:{opacity:0.8,line:{width:0.3,color:"#444"}},
    hovertext:hover(z,cc),hoverinfo:"text",
    colorbar:{title:title,thickness:12,len:0.5}};
}
function sT(d){
  return [
    {type:"scatter",x:d.sx,y:d.sy,mode:"markers",
     marker:{size:5,color:d.sc,colorscale:"Viridis",
             colorbar:{title:"log₁₀(Pop)",thickness:12,len:0.5},opacity:0.7},
     text:d.st,hoverinfo:"text"},
    {type:"scatter",x:[0,d.dxy],y:[0,d.dxy],mode:"lines",
     line:{color:"grey",dash:"dash",width:1},showlegend:false}
  ];
}
function vT(d){
  return [{type:"scatter",x:d.vx,y:d.vy,mode:"markers",
    marker:{size:4,color:"#E74C3C",opacity:0.5},text:d.vt,hoverinfo:"text"}];
}
function hT(d){
  return [
    {type:"histogram",x:d.hist.crude,nbinsx:60,name:"Crude",
     opacity:0.6,marker:{color:"#3498DB"}},
    {type:"histogram",x:d.hist.eb,nbinsx:60,name:"EB Smoothed",
     opacity:0.6,marker:{color:"#E74C3C"}}
  ];
}
function upd(i){
  const d=AD[i],b=BW[i];
  document.getElementById("bwl").textContent="Biweek: "+b;
  Plotly.react("map-crude",[mT(d.cz,d.czmax,d.cc,"Rate")],Object.assign({},ML,{title:{text:"Crude Rate — "+b,x:0.5,font:{size:14}}}));
  Plotly.react("map-eb",  [mT(d.ez,d.ezmax,d.cc,"Rate")],Object.assign({},ML,{title:{text:"EB Smoothed Rate — "+b,x:0.5,font:{size:14}}}));
  Plotly.react("scatter",sT(d),{
    title:{text:"Crude vs EB Rate (shrinkage funnel)",font:{size:13}},
    xaxis:{title:"Crude Rate (per 1k)"},yaxis:{title:"EB Rate (per 1k)"},
    margin:{l:50,r:20,t:40,b:50},height:420});
  Plotly.react("shrink",vT(d),{
    title:{text:"|Crude − EB| vs Population",font:{size:13}},
    xaxis:{title:"Population",type:"log"},yaxis:{title:"|Crude − EB| (per 1k)"},
    margin:{l:50,r:20,t:40,b:50},height:420});
  Plotly.react("hist",hT(d),{
    barmode:"overlay",
    title:{text:"Rate Distribution (positive only)",font:{size:13}},
    xaxis:{title:"Rate (per 1k)"},yaxis:{title:"Count"},
    margin:{l:50,r:20,t:40,b:50},height:420,legend:{x:0.7,y:0.95}});
}
document.getElementById("bws").addEventListener("input",function(){upd(+this.value);});
upd(0);
</script>
</body>
</html>
"""


def _build_html() -> str:
    html = _HTML
    html = html.replace("__FIRST_BW__", ALL_BIWEEKS[0])
    html = html.replace("__MAX_IDX__", str(len(ALL_BIWEEKS) - 1))
    html = html.replace("__GEOJSON__", json.dumps(GEOJSON_DATA))
    html = html.replace("__BIWEEKS__", json.dumps(ALL_BIWEEKS))
    html = html.replace("__ALL_DATA__", json.dumps(all_data))
    html = html.replace("__RATE_CS__", json.dumps(RATE_COLORSCALE))
    html = html.replace("__LOCATIONS__", json.dumps(SECTOR_IDS))
    html = html.replace("__POPULATIONS__", json.dumps(SECTOR_POPULATIONS))
    html = html.replace("__LAT__", str(MAP_CENTER_LAT))
    html = html.replace("__LON__", str(MAP_CENTER_LON))
    html = html.replace("__ZOOM__", str(MAP_ZOOM))
    return html


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Building HTML …")
    html = _build_html()
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / 1e6
    log.info("Saved → %s  (%.1f MB)", OUTPUT_HTML, size_mb)
