# =============================================================================
# CENSUS EQUIVALENCE: 2010 → 2022 CONVERSION FOR DVC PIPELINE
# =============================================================================

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path.cwd()
while (
    not (PROJECT_ROOT / "params.yaml").exists()
    and PROJECT_ROOT != PROJECT_ROOT.parent
):
    PROJECT_ROOT = PROJECT_ROOT.parent
params = yaml.safe_load(open(PROJECT_ROOT / "params.yaml"))


def _round_to_int(values: np.ndarray) -> np.ndarray:
    """Round numeric values to the nearest integer."""
    return np.floor(values + 0.5).astype(np.int64)


def _prepare_dengue_epidemic_columns(
    dengue_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build anoepid/semepid/epidemic_date like process_data_fast."""
    dengue_data = dengue_data.copy()

    if "dt_notific" not in dengue_data.columns:
        raise ValueError(
            "Dengue data must include dt_notific to derive epidemic weeks"
        )

    dengue_data["dt_notific"] = pd.to_datetime(
        dengue_data["dt_notific"],
        errors="coerce",
        format="mixed",
    )
    dengue_data = dengue_data[dengue_data["dt_notific"].notna()].copy()

    if "Dengue" in dengue_data.columns:
        dengue_data = dengue_data[dengue_data["Dengue"] != "N"].copy()

    dengue_data.drop_duplicates(inplace=True)
    dengue_data.reset_index(drop=True, inplace=True)

    dates = dengue_data["dt_notific"]
    year = np.where(dates.dt.month >= 6, dates.dt.year, dates.dt.year - 1)
    year = pd.Series(year, index=dengue_data.index)

    june_first = pd.to_datetime(year.astype(str) + "-06-01")
    offset = (june_first.dt.weekday + 1) % 7
    epi_year_start = june_first - pd.to_timedelta(offset, unit="D")

    next_june_first = pd.to_datetime((year + 1).astype(str) + "-06-01")
    next_offset = (next_june_first.dt.weekday + 1) % 7
    next_epi_year_start = next_june_first - pd.to_timedelta(
        next_offset, unit="D"
    )

    epi_year = year.copy()
    epi_year[dates >= next_epi_year_start] += 1
    dengue_data["anoepid"] = (
        epi_year.astype(str) + "_" + (epi_year + 1).astype(str).str[-2:]
    )

    week_num = ((dates - epi_year_start).dt.days // 7) + 1
    same_week_as_june1 = (dates >= epi_year_start) & (
        dates <= epi_year_start + pd.Timedelta(days=6)
    )
    week_num[same_week_as_june1] = 1
    dengue_data["semepid"] = week_num.astype(int)

    dengue_data["epidemic_date"] = dengue_data.apply(
        lambda row: f"{row['anoepid']}W{int(row['semepid']):02d}",
        axis=1,
    )

    return dengue_data


def load_epidemic_weeks_from_dengue(dengue_path: str | Path) -> list[str]:
    """Load the ordered epidemic week sequence from a dengue CSV file."""
    dengue_path = Path(dengue_path)
    if not dengue_path.exists():
        raise FileNotFoundError(
            f"Could not find dengue CSV required for interpolation: {dengue_path}"
        )

    dengue_data = pd.read_csv(dengue_path, low_memory=False)
    if "epidemic_date" not in dengue_data.columns:
        dengue_data = _prepare_dengue_epidemic_columns(dengue_data)

    epidemic_weeks = (
        pd.Index(
            dengue_data["epidemic_date"].dropna().astype(str).unique()
        )
        .sort_values()
        .tolist()
    )

    if not epidemic_weeks:
        raise ValueError("No epidemic weeks were found in dengue data")

    logger.info(
        "Loaded %s epidemic weeks from dengue data (%s to %s)",
        f"{len(epidemic_weeks):,}",
        epidemic_weeks[0],
        epidemic_weeks[-1],
    )
    return epidemic_weeks


def _week_to_ordinal(epidemic_week: str) -> int:
    """Convert epidemic week format YYYY_YYWww to an ordinal week index."""
    year_part, week_part = epidemic_week.split("W")
    start_year = int(year_part.split("_")[0])
    week_number = int(week_part)
    return (start_year * 53) + (week_number - 1)


def build_extrapolated_population_table(
    sector_comparison_df: pd.DataFrame,
    epidemic_weeks: list[str],
) -> pd.DataFrame:
    """Build a weekly population table with interpolation and extrapolation.

    The line is anchored at year 2010 (population_2010) and year 2022
    (population_2022). Weeks before 2010 and after 2022 are linearly
    extrapolated with the same slope.
    """
    if not epidemic_weeks:
        raise ValueError("epidemic_weeks must not be empty")

    anchor_2010 = _week_to_ordinal("2010_11W01")
    anchor_2022 = _week_to_ordinal("2022_23W01")
    if anchor_2022 <= anchor_2010:
        raise ValueError("Invalid interpolation anchors for 2010 and 2022")

    x_values = np.array(
        [_week_to_ordinal(week) for week in epidemic_weeks],
        dtype=np.float64,
    )
    start_values = sector_comparison_df["population_2010"].to_numpy(
        dtype=np.float64
    )
    end_values = sector_comparison_df["population_2022"].to_numpy(
        dtype=np.float64
    )
    slopes = (end_values - start_values) / (anchor_2022 - anchor_2010)

    interpolated_matrix = start_values[:, None] + slopes[:, None] * (
        x_values[None, :] - anchor_2010
    )
    interpolated_matrix = _round_to_int(interpolated_matrix)

    interpolated_df = pd.concat(
        [
            sector_comparison_df[["sector_id"]].reset_index(drop=True),
            pd.DataFrame(interpolated_matrix, columns=epidemic_weeks),
        ],
        axis=1,
    )
    return interpolated_df


def load_shapefiles():
    """Load 2010 and 2022 shapefiles for Belo Horizonte."""
    logger.info("Loading shapefiles...")

    # Load 2010 shapefiles
    shapefile_dir_2010 = Path("data/raw/IBGE/2010/shapefiles")
    bh_subdistrito_codes = [
        31062000562,
        31062000563,
        31062000564,
        31062000565,
        31062000567,
        31062000568,
        31062000569,
        31062002561,
        31062002567,
        31062006064,
        31062006066,
        31062006068,
        31062006069,
    ]

    setor_gdfs_2010 = []
    for code in bh_subdistrito_codes:
        setor_path = shapefile_dir_2010 / str(code) / f"{code}_setor.shp"
        if setor_path.exists():
            try:
                gdf = gpd.read_file(setor_path)
                gdf["Cod_subdistrito"] = code
                setor_gdfs_2010.append(gdf)
                logger.info(f"Loaded {code}: {len(gdf):,} sectors")
            except Exception as e:
                logger.warning(f"Error loading {code}: {e}")

    bh_setores_gdf = (
        pd.concat(setor_gdfs_2010, ignore_index=True)
        if setor_gdfs_2010
        else None
    )
    if bh_setores_gdf is not None:
        bh_setores_gdf = bh_setores_gdf.to_crs("EPSG:4326")
        logger.info(f"Total 2010 sectors loaded: {len(bh_setores_gdf):,}")

    # Load 2022 shapefiles
    shapefile_2022 = Path(
        "data/raw/IBGE/2022/shapefiles/MG_setores_CD2022.shp"
    )
    if shapefile_2022.exists():
        mg_setores_2022 = gpd.read_file(shapefile_2022)
        bh_sectors_gdf_2022 = mg_setores_2022[
            mg_setores_2022["CD_MUN"] == "3106200"
        ].copy()
        bh_sectors_gdf_2022 = bh_sectors_gdf_2022.to_crs("EPSG:4326")
        logger.info(f"2022 sectors loaded: {len(bh_sectors_gdf_2022):,}")
    else:
        bh_sectors_gdf_2022 = None

    return bh_setores_gdf, bh_sectors_gdf_2022


def load_population_data():
    """Load 2010 and 2022 population data."""
    logger.info("Loading population data...")

    # Load 2010 population data
    data_2010_path = Path(
        "data/raw/IBGE/2010/population/Base_informacoes_setores2010_sinopse_MG.xls"
    )

    if data_2010_path.exists():
        df_2010 = pd.read_excel(data_2010_path, sheet_name=0)
        belo_horizonte_df = df_2010[
            df_2010["Nome_do_municipio"].str.contains(
                "Belo Horizonte", case=False, na=False
            )
        ].copy()
        logger.info(
            f"2010 population data loaded: {len(belo_horizonte_df):,} sectors"
        )
    else:
        belo_horizonte_df = None

    # Load 2022 population data
    data_2022_path = Path(
        "data/raw/IBGE/2022/population/Agregados_por_setores_basico_BR_20250417.csv"
    )
    pop_2022 = {}

    if data_2022_path.exists():
        logger.info("Loading 2022 population data...")
        try:
            # Try different encodings for the CSV file
            for encoding in ["latin-1", "iso-8859-1", "cp1252", "utf-8"]:
                try:
                    df_2022 = pd.read_csv(
                        data_2022_path, sep=";", encoding=encoding
                    )
                    logger.info(
                        f"Successfully loaded 2022 data with {encoding} encoding"
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception(
                    "Could not read 2022 population file with any encoding"
                )

            # Filter for Belo Horizonte (municipality code 3106200)
            bh_2022_df = df_2022[df_2022["CD_MUN"] == 3106200].copy()
            pop_2022 = dict(
                zip(
                    bh_2022_df["CD_SETOR"].astype(str), bh_2022_df["v0001"]
                )
            )
            logger.info(
                f"2022 population data loaded: {len(pop_2022):,} sectors"
            )
        except Exception as e:
            logger.warning(f"Error loading 2022 population data: {e}")
            pop_2022 = {}
    else:
        logger.warning("2022 population data not found")
        pop_2022 = {}

    if belo_horizonte_df is not None:
        pop_2010 = dict(
            zip(
                belo_horizonte_df["Cod_setor"].astype(str),
                belo_horizonte_df["V014"],
            )
        )
        logger.info(
            f"Population mapped for {len(pop_2010):,} sectors (2010)"
        )
        return pop_2010, belo_horizonte_df, pop_2022

    return None, None, pop_2022


def perform_spatial_intersection(bh_setores_gdf, bh_sectors_gdf_2022):
    """Perform spatial intersection analysis between 2010 and 2022 sectors."""
    logger.info("Performing spatial intersection analysis...")

    # Convert to projected CRS for accurate area calculations
    logger.info(
        "Converting to projected coordinate system for spatial operations..."
    )
    bh_2010_proj = bh_setores_gdf.to_crs(
        "EPSG:31983"
    )  # SIRGAS 2000 / UTM zone 23S
    bh_2022_proj = bh_sectors_gdf_2022.to_crs("EPSG:31983")

    # Calculate original areas
    bh_2010_proj["area_2010_km2"] = bh_2010_proj.geometry.area / 1000000
    bh_2022_proj["area_2022_km2"] = bh_2022_proj.geometry.area / 1000000

    logger.info(f"2010 sectors: {len(bh_2010_proj):,}")
    logger.info(f"2022 sectors: {len(bh_2022_proj):,}")

    # Perform spatial intersection
    logger.info(
        "Computing spatial intersections (this may take a moment)..."
    )
    intersections = gpd.overlay(
        bh_2010_proj, bh_2022_proj, how="intersection"
    )

    # Calculate intersection areas
    intersections["intersection_area_km2"] = (
        intersections.geometry.area / 1000000
    )
    logger.info(f"Found {len(intersections):,} spatial overlaps")

    # Create sector linkage from spatial intersections
    sector_linkage = intersections[
        [
            "CD_GEOCODI",
            "CD_SETOR",
            "area_2010_km2",
            "area_2022_km2",
            "intersection_area_km2",
        ]
    ].copy()
    sector_linkage.columns = [
        "code_2010",
        "code_2022",
        "area_2010",
        "area_2022",
        "intersection_area",
    ]

    # Convert codes to strings and calculate fractions
    sector_linkage["code_2010"] = sector_linkage["code_2010"].astype(str)
    sector_linkage["code_2022"] = sector_linkage["code_2022"].astype(str)
    sector_linkage["fraction_of_2010"] = (
        sector_linkage["intersection_area"] / sector_linkage["area_2010"]
    )
    sector_linkage["fraction_of_2022"] = (
        sector_linkage["intersection_area"] / sector_linkage["area_2022"]
    )

    logger.info(f"All overlaps included: {len(sector_linkage):,}")
    return sector_linkage


def convert_population(sector_linkage, pop_2010, pop_2022):
    """Convert 2010 population to 2022 sector boundaries."""
    logger.info(
        "Applying spatial-based population redistribution algorithm..."
    )

    # Add population data to linkage
    sector_linkage["pop_2010"] = sector_linkage["code_2010"].map(pop_2010)
    links_with_pop = sector_linkage["pop_2010"].notna().sum()
    logger.info(
        f"Links with population data: {links_with_pop:,}/{len(sector_linkage):,} ({links_with_pop / len(sector_linkage) * 100:.1f}%)"
    )

    # Initialize results
    pop_2022_equivalent = {}
    conversion_log = []

    logger.info(
        f"Processing {len(sector_linkage):,} spatial relationships"
    )

    # Group by 2022 sectors to accumulate population
    for code_2022 in sector_linkage["code_2022"].unique():
        sector_relationships = sector_linkage[
            sector_linkage["code_2022"] == code_2022
        ]

        total_pop_for_2022 = 0
        contributing_2010_sectors = []

        for _, relationship in sector_relationships.iterrows():
            code_2010 = relationship["code_2010"]
            fraction_of_2010 = relationship["fraction_of_2010"]
            pop_2010_val = relationship["pop_2010"]

            if pd.notna(pop_2010_val):
                contributed_pop = pop_2010_val * fraction_of_2010
                total_pop_for_2022 += contributed_pop

                contributing_2010_sectors.append(
                    {
                        "code_2010": code_2010,
                        "fraction": fraction_of_2010,
                        "contributed_pop": contributed_pop,
                        "original_pop": pop_2010_val,
                    }
                )

        if total_pop_for_2022 > 0:
            pop_2022_equivalent[code_2022] = total_pop_for_2022

            # Determine conversion type
            if len(contributing_2010_sectors) == 1:
                contrib = contributing_2010_sectors[0]
                if contrib["fraction"] >= 0.8:
                    conversion_type = "maintained"
                    method = f"direct_transfer_fraction_{contrib['fraction']:.3f}"
                else:
                    conversion_type = "subdivided"
                    method = f"partial_transfer_fraction_{contrib['fraction']:.3f}"
            else:
                conversion_type = "aggregated"
                method = (
                    f"sum_from_{len(contributing_2010_sectors)}_sectors"
                )

            source_codes = [
                c["code_2010"] for c in contributing_2010_sectors
            ]
            original_total = sum(
                c["original_pop"] for c in contributing_2010_sectors
            )

            conversion_log.append(
                {
                    "type": conversion_type,
                    "code_2010": "|".join(source_codes),
                    "code_2022": code_2022,
                    "pop_original": original_total,
                    "pop_converted": total_pop_for_2022,
                    "method": method,
                    "num_sources": len(contributing_2010_sectors),
                }
            )

    # Summary statistics
    conversion_df = pd.DataFrame(conversion_log)
    type_counts = conversion_df["type"].value_counts()

    logger.info("Conversion results:")
    for conv_type, count in type_counts.items():
        total_pop = conversion_df[conversion_df["type"] == conv_type][
            "pop_converted"
        ].sum()
        logger.info(
            f"  {conv_type.title():>12}: {count:,} sectors ({total_pop:,.0f} people)"
        )

    # Conservation statistics
    unique_2010_pop = (
        sector_linkage.groupby("code_2010")["pop_2010"].first().sum()
    )
    converted_total = sum(pop_2022_equivalent.values())
    conservation_rate = converted_total / unique_2010_pop * 100

    logger.info("Overall conversion statistics:")
    logger.info(
        f"  Original 2010 population (unique sectors): {unique_2010_pop:,.0f}"
    )
    logger.info(f"  Converted 2022 equivalent: {converted_total:,.0f}")
    logger.info(f"  Conservation rate: {conservation_rate:.2f}%")

    # Create output DataFrames
    equivalent_population_2022 = pd.DataFrame(
        [
            {"CD_SETOR": k, "pop_2010_equivalent": v}
            for k, v in pop_2022_equivalent.items()
        ]
    )

    return equivalent_population_2022, conversion_df, pop_2022_equivalent


def main():
    """Main function to execute the census equivalence conversion."""
    logger.info("=" * 80)
    logger.info("CENSUS EQUIVALENCE: 2010 → 2022 CONVERSION")
    logger.info("=" * 80)
    logger.info(
        "Converting 2010 population data to 2022 sector boundaries"
    )
    logger.info(
        "Using spatial intersection analysis and area-based distribution"
    )
    logger.info("=" * 80)

    try:
        # Execute the conversion
        bh_setores_gdf, bh_sectors_gdf_2022 = load_shapefiles()

        if bh_setores_gdf is not None and bh_sectors_gdf_2022 is not None:
            pop_2010, belo_horizonte_df, pop_2022 = load_population_data()

            if pop_2010 is not None:
                sector_linkage = perform_spatial_intersection(
                    bh_setores_gdf, bh_sectors_gdf_2022
                )
                (
                    equivalent_population_2022,
                    conversion_df,
                    pop_2022_equivalent,
                ) = convert_population(sector_linkage, pop_2010, pop_2022)

                # Create output directory
                output_dir = Path("data/processed")
                output_dir.mkdir(exist_ok=True)

                # Save results
                logger.info("Saving conversion results...")

                # Save the sector equivalence mapping (2010 → 2022)
                logger.info(
                    "Saving sector equivalence mapping (2010 → 2022)..."
                )
                equivalence_mapping = sector_linkage[
                    [
                        "code_2010",
                        "code_2022",
                        "fraction_of_2010",
                        "fraction_of_2022",
                        "intersection_area",
                    ]
                ].copy()
                equivalence_mapping.to_csv(
                    output_dir / "sector_equivalence_2010_to_2022.csv",
                    index=False,
                )
                logger.info(
                    f"Saved sector equivalence mapping: {len(equivalence_mapping):,} relationships"
                )

                # Create the required CSV with sector ID, 2010 population, and 2022 population
                logger.info("Creating sector population comparison CSV...")

                # Get all unique 2022 sectors
                all_2022_sectors = (
                    bh_sectors_gdf_2022["CD_SETOR"].astype(str).tolist()
                )

                # Create the output DataFrame
                sector_comparison = []
                for sector_id in all_2022_sectors:
                    pop_2010_equiv = round(
                        pop_2022_equivalent.get(sector_id, 0)
                    )  # Round 2010 population to integer
                    pop_2022_actual = pop_2022.get(
                        sector_id, 0
                    )  # Actual 2022 population (already integer)

                    sector_comparison.append(
                        {
                            "sector_id": sector_id,
                            "population_2010": pop_2010_equiv,
                            "population_2022": pop_2022_actual,
                        }
                    )

                # Create DataFrame and save CSV
                sector_comparison_df = pd.DataFrame(sector_comparison)
                sector_comparison_df.to_csv(
                    output_dir / "population_data.csv", index=False
                )
                logger.info(
                    f"Saved sector population comparison CSV: {len(sector_comparison_df):,} sectors"
                )

                # Build and save weekly interpolated/extrapolated population table
                epidemic_weeks = load_epidemic_weeks_from_dengue(
                    dengue_path=params["all"]["paths"]["data"][
                        "processed"
                    ]["dengue"]
                )
                interpolated_population_df = (
                    build_extrapolated_population_table(
                        sector_comparison_df,
                        epidemic_weeks,
                    )
                )
                interpolated_population_df.to_csv(
                    output_dir / "interpolated_population_data.csv",
                    index=False,
                )
                logger.info(
                    "Saved interpolated/extrapolated population CSV: %s sectors x %s weeks",
                    f"{len(interpolated_population_df):,}",
                    f"{len(epidemic_weeks):,}",
                )

                # Save the GeoJSON of 2022 sectors with both populations
                bh_sectors_with_both_pop = bh_sectors_gdf_2022.copy()
                bh_sectors_with_both_pop["pop_2010"] = (
                    bh_sectors_with_both_pop["CD_SETOR"]
                    .astype(str)
                    .map(pop_2022_equivalent)
                    .fillna(0)
                    .round()
                    .astype(int)
                )
                bh_sectors_with_both_pop["pop_2022"] = (
                    bh_sectors_with_both_pop["CD_SETOR"]
                    .astype(str)
                    .map(pop_2022)
                    .fillna(0)
                    .astype(int)
                )

                # Save as GeoJSON
                bh_sectors_with_both_pop.to_file(
                    output_dir
                    / "bh_sectors_2022_with_populations.geojson",
                    driver="GeoJSON",
                )
                logger.info(
                    f"Saved 2022 sectors GeoJSON with both populations: {len(bh_sectors_with_both_pop):,} sectors"
                )

                # Print summary statistics
                total_pop_2010 = sector_comparison_df[
                    "population_2010"
                ].sum()
                total_pop_2022 = sector_comparison_df[
                    "population_2022"
                ].sum()
                sectors_with_2010_data = (
                    sector_comparison_df["population_2010"] > 0
                ).sum()
                sectors_with_2022_data = (
                    sector_comparison_df["population_2022"] > 0
                ).sum()

                logger.info("Summary statistics:")
                logger.info(
                    f"  Total 2010 population (equivalent): {total_pop_2010:,}"
                )
                logger.info(
                    f"  Total 2022 population (actual): {total_pop_2022:,}"
                )
                logger.info(
                    f"  Sectors with 2010 data: {sectors_with_2010_data:,} / {len(sector_comparison_df):,}"
                )
                logger.info(
                    f"  Sectors with 2022 data: {sectors_with_2022_data:,} / {len(sector_comparison_df):,}"
                )

                logger.info(
                    "Census equivalence conversion completed successfully!"
                )
                logger.info(
                    f"Conservation rate: {conversion_df['pop_converted'].sum() / sector_linkage.groupby('code_2010')['pop_2010'].first().sum() * 100:.2f}%"
                )

            else:
                logger.error("Cannot proceed without population data")
                raise Exception("Population data not available")
        else:
            logger.error("Cannot proceed without shapefile data")
            raise Exception("Shapefile data not available")

    except Exception as e:
        logger.error(f"Census equivalence conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
