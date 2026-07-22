"""SKATER-CV RMSE choropleth — one interactive HTML file.

dashboard_rmse_map.html
  Choropleth map of BH census sectors coloured by held-out MLP RMSE of
  the region (SKATER cluster) each sector belongs to.  A dropdown
  selects which leave-one-epidemic-year-out fold to inspect (each fold
  has its own SKATER partition — geometry differs across folds even at
  the same C); a slider then sweeps that fold's available C values.
  Colour scale (RMSE) is fixed globally across every fold/C combination
  so shades are directly comparable.  Clicking a sector shows that
  region's predicted-vs-actual EB rate over the held-out epidemic year
  (small matplotlib PNG, pre-rendered — keeps the interactive Plotly
  side lightweight) in a side panel.

Both the source partitions/metrics and this script's own output are
namespaced by params.yaml[skater].run_label — same convention as the
all-years `skater` stage — so it always renders whichever run is
currently configured (e.g. one S_min value of a sweep).

Inputs:
  results/multiscale/partitions/<run_label>/fold_<year>/cluster_assignments.csv
  results/multiscale/skater_cv/<run_label>/metrics_skater.csv
  results/multiscale/skater_cv/<run_label>/predictions.csv
  data/dvc/process_population_data/bh_sectors_2022_with_populations.geojson
Outputs:
  results/multiscale/skater_cv/<run_label>/rmse_map.html
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────
PART_BASE = Path("results/multiscale/partitions")
SKATER_CV_BASE = Path("results/multiscale/skater_cv")
GEOJSON_PATH = Path(
    "data/dvc/process_population_data/"
    "bh_sectors_2022_with_populations.geojson"
)
BH_CENTER = {"lat": -19.917, "lon": -43.934}


# ── Data loaders ──────────────────────────────────────────────────────

def _load_run_label() -> str:
    """Read params.yaml[skater].run_label — same source as the all-years
    dashboard, so this script always targets the currently configured run.
    """
    with open("params.yaml") as fh:
        return yaml.safe_load(fh)["skater"]["run_label"]


def _load_geojson() -> dict:
    """Load the BH census sector GeoJSON FeatureCollection."""
    with open(GEOJSON_PATH) as fh:
        return json.load(fh)


def _load_metrics(cv_dir: Path) -> pd.DataFrame:
    """Load per-(fold, C, region) test metrics with cluster_id parsed out.

    `unit` is formatted `C{C}__c{cluster_id}` by
    `src.multiscale.skater_cv_models`; cluster_id is not a separate
    column so it is parsed back out here.

    Args:
        cv_dir: results/multiscale/skater_cv/<run_label>/.

    Returns:
        metrics_skater.csv rows plus an added integer `cluster_id` col.
    """
    df = pd.read_csv(cv_dir / "metrics_skater.csv")
    df["cluster_id"] = (
        df["unit"].str.split("__c").str[1].astype(int)
    )
    return df


def _load_predictions(cv_dir: Path) -> pd.DataFrame:
    """Load raw held-out predictions with cluster_id parsed out.

    Args:
        cv_dir: results/multiscale/skater_cv/<run_label>/.

    Returns:
        predictions.csv rows [unit, fold_year, C, biweek, y_true, y_pred]
        plus an added integer `cluster_id` column (see `_load_metrics`).
    """
    df = pd.read_csv(cv_dir / "predictions.csv")
    df["cluster_id"] = df["unit"].str.split("__c").str[1].astype(int)
    return df


def _image_key(fold_year: str, c_value: int, cluster_id: int) -> str:
    """Build the lookup key shared by customdata and the IMAGES map."""
    return f"{fold_year}|{c_value}|{cluster_id}"


def _load_fold_assignments(part_dir: Path, fold_year: str) -> pd.DataFrame:
    """Load one fold's sector → cluster_id assignments, all C values.

    Args:
        part_dir:  results/multiscale/partitions/<run_label>/.
        fold_year: e.g. '2015_16'.

    Returns:
        DataFrame [C, sector_id, cluster_id].
    """
    path = part_dir / f"fold_{fold_year}" / "cluster_assignments.csv"
    return pd.read_csv(path, dtype={"sector_id": str})


# ── Predict-vs-target panel images ──────────────────────────────────────

def _render_prediction_png(group: pd.DataFrame, title: str) -> str:
    """Render one region-fold's actual-vs-predicted time series as a PNG.

    Uses matplotlib (not Plotly) so the image is a small raster blob
    instead of another interactive trace — keeps the ~30-region panel
    gallery from bloating the already-large map HTML.

    Args:
        group: Rows for one (fold, C, cluster) [biweek, y_true, y_pred],
               any order.
        title: Short title drawn above the plot.

    Returns:
        A `data:image/png;base64,...` URI string.
    """
    sub = group.sort_values("biweek")
    fig, ax = plt.subplots(figsize=(4.2, 2.3), dpi=90)
    ax.plot(sub["biweek"], sub["y_true"], label="actual", linewidth=1.5)
    ax.plot(
        sub["biweek"], sub["y_pred"], label="predicted",
        linewidth=1.5, linestyle="--",
    )
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("EB rate", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xticks(ax.get_xticks()[::max(1, len(sub) // 6)])
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_prediction_images(predictions: pd.DataFrame) -> dict[str, str]:
    """Pre-render every (fold, C, cluster) predict-vs-target PNG.

    Args:
        predictions: Output of `_load_predictions` — [unit, fold_year, C,
                     cluster_id, biweek, y_true, y_pred].

    Returns:
        Mapping `_image_key(fold_year, C, cluster_id)` → PNG data URI.
    """
    images: dict[str, str] = {}
    for (fold_year, c_value, cid), group in predictions.groupby(
        ["fold_year", "C", "cluster_id"]
    ):
        title = f"{fold_year} — C={c_value} — cluster {cid}"
        images[_image_key(fold_year, c_value, cid)] = _render_prediction_png(
            group, title
        )
    logger.info("Rendered %d predict-vs-target PNGs", len(images))
    return images


# ── Per-(fold, C) frame construction ────────────────────────────────────

def _frame_for_c(
    fold_year: str,
    c_value: int,
    asgn_c: pd.DataFrame,
    metrics_c: pd.DataFrame,
    sector_order: list[str],
) -> tuple[list[float], list[str], list[str]]:
    """Build the RMSE z-array, hover text and click keys for one frame.

    Args:
        fold_year:    e.g. '2015_16' — used to build the click image key.
        c_value:      Number of clusters in this frame.
        asgn_c:       Assignments filtered to one C [sector_id, cluster_id].
        metrics_c:    Metrics filtered to the same fold and C
                      [cluster_id, mlp_rmse, mlp_mae, mlp_r2, mlp_spearman,
                      naive_rmse, n_sectors, pop].
        sector_order: Sector IDs in the order GeoJSON features appear.

    Returns:
        z:          RMSE per sector, aligned to sector_order.
        text:       Hover text per sector, aligned to sector_order.
        customdata: Predict-vs-target image key per sector ("" if no
                    data), aligned to sector_order — read by the click
                    handler to look up the panel PNG.
    """
    cluster_of = asgn_c.set_index("sector_id")["cluster_id"]
    stats = metrics_c.set_index("cluster_id")

    z: list[float] = []
    text: list[str] = []
    customdata: list[str] = []
    for sector_id in sector_order:
        cid = int(cluster_of.get(sector_id, -1))
        if cid not in stats.index:
            z.append(float("nan"))
            text.append(f"<b>{sector_id}</b><br>no data")
            customdata.append("")
            continue
        row = stats.loc[cid]
        z.append(float(row["mlp_rmse"]))
        text.append(
            f"<b>{sector_id}</b><br>"
            f"cluster {cid}<br>"
            f"RMSE={row['mlp_rmse']:.3f}<br>"
            f"MAE={row['mlp_mae']:.3f}<br>"
            f"R²={row['mlp_r2']:.3f}<br>"
            f"Spearman r={row['mlp_spearman']:.3f}<br>"
            f"naive RMSE={row['naive_rmse']:.3f}<br>"
            f"sectors={int(row['n_sectors'])}<br>"
            f"pop={row['pop']:.0f}<br>"
            f"<i>click for predict-vs-actual</i>"
        )
        customdata.append(_image_key(fold_year, c_value, cid))
    return z, text, customdata


# ── Figure assembly ──────────────────────────────────────────────────

def build_figure(
    part_dir: Path,
    metrics: pd.DataFrame,
    geojson: dict,
) -> go.Figure:
    """Build the fold-dropdown + C-slider RMSE choropleth figure.

    Each fold gets its own slider (steps = that fold's available C
    values); only the active fold's slider is visible at a time,
    toggled by the dropdown.  The dropdown also resets the choropleth
    to that fold's first C.  The colour scale is fixed globally
    (`metrics['mlp_rmse'].min()/.max()`) so shades stay comparable
    across every fold/C combination.

    Args:
        part_dir: results/multiscale/partitions/<run_label>/.
        metrics:  Full metrics_skater.csv + parsed cluster_id.
        geojson:  GeoJSON FeatureCollection for BH sectors.

    Returns:
        go.Figure ready to write as standalone HTML.
    """
    folds = sorted(metrics["fold_year"].unique())
    sector_order = [
        str(f["properties"]["CD_SETOR"]) for f in geojson["features"]
    ]
    # Clip the colour scale at 5 — a handful of unstable high-C region
    # fits (esp. small S_min) blow up to much higher RMSE and would
    # otherwise wash out contrast across the typical ~1.5-2.5 range.
    # Values above 5 still render, just clamped to the top colour.
    rmse_min = float(metrics["mlp_rmse"].min())
    rmse_max = min(5.0, float(metrics["mlp_rmse"].max()))

    # ── Precompute every (fold, C) frame ───────────────────────────────
    fold_c_values: dict[str, list[int]] = {}
    frames: dict[tuple[str, int], tuple[list[float], list[str], list[str]]] = {}
    for fold_year in folds:
        asgn = _load_fold_assignments(part_dir, fold_year)
        fold_metrics = metrics[metrics["fold_year"] == fold_year]
        c_values = sorted(fold_metrics["C"].unique().tolist())
        fold_c_values[fold_year] = c_values
        for c_value in c_values:
            asgn_c = asgn[asgn["C"] == c_value]
            metrics_c = fold_metrics[fold_metrics["C"] == c_value]
            frames[(fold_year, c_value)] = _frame_for_c(
                fold_year, c_value, asgn_c, metrics_c, sector_order
            )

    fold0 = folds[0]
    c0 = fold_c_values[fold0][0]
    z0, text0, customdata0 = frames[(fold0, c0)]

    # ── Fixed map traces ──────────────────────────────────────────────
    choro = go.Choroplethmap(
        geojson=geojson,
        locations=sector_order,
        z=z0,
        text=text0,
        customdata=customdata0,
        hovertemplate="%{text}<extra></extra>",
        featureidkey="properties.CD_SETOR",
        colorscale="RdYlGn_r",
        zmin=rmse_min,
        zmax=rmse_max,
        marker_opacity=0.75,
        marker_line_width=0.1,
        showscale=True,
        colorbar={"title": {"text": "RMSE"}},
        name="RMSE",
    )

    # ── One slider per fold, only the active fold's slider visible ────
    def _slider_for_fold(fold_year: str, visible: bool) -> dict:
        steps = []
        for c_value in fold_c_values[fold_year]:
            z, text, customdata = frames[(fold_year, c_value)]
            steps.append({
                "label": str(c_value),
                "method": "restyle",
                "args": [
                    {"z": [z], "text": [text], "customdata": [customdata]},
                    [0],
                ],
            })
        return {
            "active": 0,
            "steps": steps,
            "visible": visible,
            "x": 0.05,
            "len": 0.9,
            "y": 0.02,
            "yanchor": "top",
            "currentvalue": {
                "prefix": f"{fold_year} — C = ",
                "visible": True,
                "xanchor": "center",
            },
        }

    sliders_all = [
        _slider_for_fold(fold_year, visible=(fold_year == fold0))
        for fold_year in folds
    ]

    # ── Dropdown: switch fold → reset map to that fold's first C and
    # swap which slider is visible ─────────────────────────────────────
    # Bracket-path relayout keys ("sliders[i].visible") toggle visibility
    # without re-serialising every slider's full step data per button —
    # re-sending the whole `sliders` array 3x (once per button) would
    # quadruple the embedded frame data and bloat the HTML.
    buttons = []
    for i, fold_year in enumerate(folds):
        c_first = fold_c_values[fold_year][0]
        z, text, customdata = frames[(fold_year, c_first)]
        layout_upd = {
            f"sliders[{j}].visible": (j == i) for j in range(len(folds))
        }
        buttons.append({
            "label": fold_year,
            "method": "update",
            "args": [
                {"z": [z], "text": [text], "customdata": [customdata]},
                layout_upd,
                [0],
            ],
        })

    fig = go.Figure(data=[choro])
    fig.update_layout(
        map={
            "style": "open-street-map",
            "zoom": 11,
            "center": BH_CENTER,
            "domain": {"x": [0, 1], "y": [0.10, 1.0]},
        },
        sliders=sliders_all,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 0.01,
            "y": 1.08,
            "xanchor": "left",
            "yanchor": "top",
            "showactive": True,
        }],
        height=950,
        margin={"t": 60, "b": 20, "l": 50, "r": 20},
        title={
            "text": f"SKATER-CV — held-out region RMSE — {fold0}, C={c0}",
            "x": 0.5,
        },
    )
    return fig


# ── Page assembly ─────────────────────────────────────────────────────

_CLICK_JS = """
<script>
const mapDiv = document.getElementById('rmse-map');
const panelImg = document.getElementById('predict-img');
const panelCaption = document.getElementById('predict-caption');
mapDiv.on('plotly_click', function(evt) {
  const pt = evt.points[0];
  // Choroplethmap click events don't reliably echo a restyled
  // `customdata` in evt.points[].customdata (stale after slider/dropdown
  // restyles even though the map colour/hover — driven by the same
  // restyle — updates correctly). Read the live trace data instead,
  // which restyle always keeps in sync.
  const liveCustomdata = mapDiv.data && mapDiv.data[0] &&
    mapDiv.data[0].customdata;
  const key = pt && liveCustomdata && liveCustomdata[pt.pointIndex];
  if (!key || !(key in IMAGES)) {
    panelCaption.textContent = 'No prediction data for this sector.';
    panelImg.style.display = 'none';
    return;
  }
  const [foldYear, cValue, cluster] = key.split('|');
  panelImg.src = IMAGES[key];
  panelImg.style.display = 'block';
  panelCaption.textContent =
    `Fold ${foldYear} — C=${cValue} — cluster ${cluster}`;
});
</script>
"""


def _assemble_page(fig: go.Figure, images: dict[str, str]) -> str:
    """Wrap the map figure with a click-driven predict-vs-target panel.

    Args:
        fig:    The RMSE choropleth figure from build_figure().
        images: `_image_key(...)` → PNG data URI, from
                _build_prediction_images().

    Returns:
        Full standalone HTML page.
    """
    map_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="rmse-map"
    )
    images_json = json.dumps(images)

    return "\n".join([
        "<!DOCTYPE html><html>",
        "<head><meta charset='utf-8'>",
        "<title>SKATER-CV RMSE map</title>",
        "<style>",
        "body{font-family:sans-serif;margin:0;padding:12px}",
        ".layout{display:flex;gap:16px;align-items:flex-start}",
        "#map-col{flex:3;min-width:0}",
        "#panel-col{flex:1;min-width:280px;position:sticky;top:12px;"
        "border:1px solid #ccc;border-radius:6px;padding:10px}",
        "#predict-img{max-width:100%;display:none}",
        "#predict-caption{font-size:13px;color:#333;margin-top:6px}",
        "</style>",
        "</head><body>",
        "<div class='layout'>",
        f"<div id='map-col'>{map_html}</div>",
        "<div id='panel-col'>",
        "<h3>Predict vs actual</h3>",
        "<p id='predict-caption'>Click a sector on the map.</p>",
        "<img id='predict-img'>",
        "</div>",
        "</div>",
        f"<script>const IMAGES = {images_json};</script>",
        _CLICK_JS,
        "</body></html>",
    ])


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    """Build and write the SKATER-CV RMSE choropleth HTML."""
    run_label = _load_run_label()
    part_dir = PART_BASE / run_label
    cv_dir = SKATER_CV_BASE / run_label
    out_path = cv_dir / "rmse_map.html"

    logger.info("Loading metrics, geojson, predictions (run_label=%s)", run_label)
    metrics = _load_metrics(cv_dir)
    geojson = _load_geojson()
    predictions = _load_predictions(cv_dir)

    fig = build_figure(part_dir, metrics, geojson)
    images = _build_prediction_images(predictions)
    page = _assemble_page(fig, images)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    logger.info("RMSE map → %s", out_path)


if __name__ == "__main__":
    main()
