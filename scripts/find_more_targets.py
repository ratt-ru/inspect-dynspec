from astropy.coordinates import SkyCoord, Angle
from requests.exceptions import ConnectionError, Timeout
from astroquery.gaia import Gaia
from astropy import units as u
from astroquery.simbad import Simbad
import pandas as pd
import numpy as np
import os


def read_existing_ecsv(ecsv_path):
    try:
        df = pd.read_csv(ecsv_path, sep=r"\s+", comment="#")
        expected_cols = [
            "id",
            "did",
            "cid",
            "pid",
            "x",
            "y",
            "pos.ra",
            "pos.dec",
            "stokes",
        ]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in ECSV: {missing}")
        return df
    except Exception as e:
        print(f"Could not read ECSV file: {e}")
        return pd.DataFrame(
            columns=["id", "did", "cid", "pid", "x", "y", "pos.ra", "pos.dec", "stokes"]
        )


def write_final_ecsv(df, ecsv_path):
    header = "id did cid pid x y pos.ra pos.dec stokes"
    with open(ecsv_path, "w") as f:
        f.write(header + "\n")
        for _, row in df.iterrows():
            f.write(
                f"{row['id']} {row['did']} {row['cid']} {row['pid']} {row['x']} {row['y']} {row['pos.ra']} {row['pos.dec']} {row['stokes']}\n"
            )


def query_Gaia_ID(sky_coordinates, radius=0.001, id_version="dr3") -> str:
    """
    Query Gaia dr2, edr3 or dr3 (specify with id_version) for the gaia id given the right ascension and
    declination of the anticipated source. Form a circle of radius `radius` and collect
    all sources inside of it.
    Args:
        sky_coordinates: SkyCoord of the star to search for.
        radius: Radius in degrees to search around the provided coordinates. Default is 0.001.
        id_version: Version of the Gaia ID to fetch. Default is 'dr3'.
    Returns:
        gaia_id: Gaia ID of the star found, None if none is found.
    """
    if sky_coordinates is None:
        return None
    else:
        if id_version == "dr3":
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        else:
            Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

        gaia_job = Gaia.cone_search_async(
            sky_coordinates, radius=u.Quantity(radius, u.deg)
        )
        gaia_data = gaia_job.get_results()
        if len(gaia_data) > 0:
            # returns smallest distance from source first - which is appropriate for us
            return gaia_data
        else:
            return None


def query_Simbad_for_Coordinates(object_name: str) -> SkyCoord:
    """
    Query SIMBAD on a star name (str). If the returned data is empty, return None,
    else return the SkyCoordinates for that star.
    Args:
        object_name: Name of the star to search for.
    Returns:
        simbad_coord: SkyCoord of the star found, None if none is found.
    """
    try:
        simbad_table = Simbad.query_object(object_name)
    except ValueError:
        simbad_table = None
    except (ConnectionError, Timeout):
        print("A network occurred while querying Simbad.")
        return None
    try:
        if simbad_table:
            if simbad_table["RA"] and simbad_table["DEC"]:
                simbad_coord = SkyCoord(
                    ra=float(Angle(simbad_table["RA"], unit=u.hourangle).degree[0]),
                    dec=float(Angle(simbad_table["DEC"], unit=u.deg).degree[0]),
                    unit=(u.deg, u.deg),
                )
                return simbad_coord
            else:
                return None
        else:
            return None
    except Exception as e:
        print(
            f"""Simbad returned:
            RA: {simbad_table['RA']}, DEC: {simbad_table['DEC']}
            while fetching Simbad coordinates for star: {object_name}.
            Encountered exception: {e}"""
        )
        return None


def find_targets_in_csv_catalog(catalog_path, ra, dec, radius_deg, max_distance=None):
    """
    Find all targets in the catalog within a given radius (degrees) of (ra, dec).
    Optionally filter by max_distance (parsecs).
    Returns a DataFrame with star_name, ra, dec, gaia_dr3, star_distance.
    """
    # Load catalog
    df = pd.read_csv(catalog_path)

    # Drop rows without coords
    df = df.dropna(subset=["ra", "dec", "star_name", "gaia_dr3", "star_distance"])

    # Create SkyCoord objects for catalog and input
    catalog_coords = SkyCoord(
        ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs"
    )
    input_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Calculate separations
    sep = input_coord.separation(catalog_coords).deg

    # Filter by radius
    mask = sep <= radius_deg
    filtered = df[mask]

    if max_distance is not None:
        filtered = filtered[filtered["star_distance"] <= max_distance]

    result = filtered[
        ["star_name", "ra", "dec", "gaia_dr3", "star_distance"]
    ].reset_index(drop=True)
    return result


def find_more_targets(
    ecsv: str,
    ra: float,
    dec: float,
    radius: float = 1,
    distance: float = 200,
    identifier: str = "",
    tolerance: float = 1e-3,
) -> str:
    """
    Find more targets for dynamic spectra analysis, and append them to a provided ecsv file.
    If no ecsv file is provided, a new one will be created.
    Args:
        ecsv: Path to the ecsv file to append targets to. If not provided,
              a new ecsv file will be created.
        ra: Right Ascension of the Measurement Set phase center in degrees.
        dec: Declination of the Measurement Set phase center in degrees.
        radius: Radius in degrees to search around the provided coordinates. Default is 1.
        distance: Distance in parsecs to search for targets. Default is 200.
        identifier: Identifier name for the target ecsv file - will be used in the output filename
        tolerance: Tolerance for RA/Dec matching when replacing EU catalog entries with Gaia entries (default: 1e-3 degrees)
    Returns:
        ecsv: Path to the ecsv file with the appended targets.
    """
    sky_coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Query Gaia for the closest source
    try:
        gaia_sources = query_Gaia_ID(sky_coordinates, radius=radius)
        if gaia_sources is not None:
            print(f"Found {len(gaia_sources)} targets in the gaia catalog within the specified radius.")
        else:
            print("No Gaia sources found within the specified radius.")
    except Exception as e:
        print(f"An error occurred while querying Gaia: {e}")

    try:
        eu_exoplanet = find_targets_in_csv_catalog(
            "/home/myburgh/exoplanet_eu_catalog_dr3.csv",
            ra=ra,
            dec=dec,
            radius_deg=radius,
            max_distance=distance,
        )
        print(
            f"Found {len(eu_exoplanet)} targets in the eu exoplanet catalog within the specified radius."
        )
    except Exception as e:
        print(f"An error occurred while querying the exoplanet catalog: {e}")

    if os.path.exists(ecsv):
        orig_df = read_existing_ecsv(ecsv)
        orig_targets = pd.DataFrame({
            "star_name": orig_df["id"],
            "ra": orig_df["pos.ra"],
            "dec": orig_df["pos.dec"],
            "gaia_dr3": orig_df["id"].apply(lambda x: x.split(":")[-1] if ":" in str(x) else str(x)),
            "star_distance": np.nan,
        })
    else:
        orig_targets = pd.DataFrame(columns=["star_name", "ra", "dec", "gaia_dr3", "star_distance"])

    if len(eu_exoplanet) == 0:
        target_df = orig_targets.copy()
    else:
        target_df = pd.concat([orig_targets, eu_exoplanet.copy()], ignore_index=True)

    if gaia_sources is not None:
        for gaia_source in gaia_sources:
            gaia_dr3_id = gaia_source["source_id"]
            try:
                parallax = gaia_source["parallax"]
            except Exception:
                parallax = np.nan
            try:
                star_distance = 1000.0 / parallax if parallax > 0 else np.nan
            except Exception:
                star_distance = np.nan

            if not np.isnan(star_distance) and star_distance <= distance:
                print(
                        f"Gaia source {gaia_source['source_id']} found, at distance ({star_distance:.2f} pc). Appending."
                    )
                try:
                    ra_val = gaia_source["ra"]
                    dec_val = gaia_source["dec"]
                except Exception:
                    ra_val = np.nan
                    dec_val = np.nan
                sol_id = gaia_source["solution_id"]

                # Replace close EU catalog sources with Gaia ones
                close_mask = (abs(target_df["ra"] - ra_val) < tolerance) & (
                    abs(target_df["dec"] - dec_val) < tolerance
                )
                if close_mask.any():
                    for idx in target_df[close_mask].index:
                        eu_row = target_df.loc[idx]
                        print(
                            f"Replacing EU catalog entry (id={eu_row.get('star_name', 'N/A')}, ra={eu_row['ra']}, dec={eu_row['dec']}) "
                            f"with Gaia entry (id={sol_id}, gaia_dr3={gaia_dr3_id}, ra={ra_val}, dec={dec_val}) "
                            f"due to close RA/Dec match (tolerance={tolerance})."
                        )
                    target_df = target_df[~close_mask]
                if gaia_dr3_id not in target_df["gaia_dr3"].values:
                    new_row = {
                        "star_name": sol_id,
                        "ra": ra_val,
                        "dec": dec_val,
                        "gaia_dr3": gaia_dr3_id,
                        "star_distance": star_distance,
                    }
                    new_row_df = pd.DataFrame([new_row])
                    if target_df.empty:
                        target_df = new_row_df
                    else:
                        target_df = pd.concat([target_df, new_row_df], ignore_index=True)
            else:
                if not np.isnan(star_distance):
                    print(
                        f"Gaia source {gaia_source['source_id']} found, but its distance ({star_distance:.2f} pc) exceeds the specified limit ({distance} pc). Not appending."
                    )
    else:
        print("No Gaia source found within the specified radius.")

    new_rows = []
    orig_ids = set(orig_df["id"]) if os.path.exists(ecsv) else set()
    for idx, row in target_df.iterrows():
        if (
            row["star_name"] in orig_ids
            or ("id" in row and row["star_name"] == row["id"])
        ):
            id_val = row["star_name"]
        else:
            id_val = f"{row['star_name']}:{row['gaia_dr3']}"
        new_rows.append(
            {
                "id": id_val,
                "did": "0",
                "cid": "0",
                "pid": "0",
                "x": "0",
                "y": "0",
                "pos.ra": row["ra"],
                "pos.dec": row["dec"],
                "stokes": "I",
            }
        )

    if os.path.exists(ecsv):
        existing_df = read_existing_ecsv(ecsv)
        ecsv_dir = os.path.dirname(os.path.abspath(ecsv))
        if identifier:
            output_ecsv = os.path.join(ecsv_dir, f"{identifier}_targets.ecsv")
        else:
            output_ecsv = os.path.join(ecsv_dir, "targets_appended.ecsv")
    else:
        existing_df = pd.DataFrame(
            columns=["id", "did", "cid", "pid", "x", "y", "pos.ra", "pos.dec", "stokes"]
        )
        if identifier:
            output_ecsv = f"{identifier}_targets.ecsv"
        else:
            output_ecsv = "targets.ecsv"

    final_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["id"])

    # Write out the final ECSV
    write_final_ecsv(final_df, output_ecsv)
    print(f"Final ECSV written to {output_ecsv}")
    return output_ecsv


if __name__ == "__main__":
    import argparse
    import os

    args = argparse.ArgumentParser(
        description="Find more targets for dynamic spectra analysis, and append them to a provided ecsv (when provided)."
    )
    args.add_argument(
        "--ecsv",
        type=str,
        default=None,
        required=False,
        help="Path to the ecsv file to append targets to. If not provided, a new ecsv file will be created.",
    )
    args.add_argument(
        "--ra",
        type=float,
        required=True,
        help="Right Ascension of the Measurement Set phase center in degrees.",
    )
    args.add_argument(
        "--dec",
        type=float,
        required=True,
        help="Declination of the Measurement Set phase center in degrees.",
    )
    args.add_argument(
        "--radius",
        type=float,
        default=1,
        help="Radius in degrees to search around the provided coordinates. Default is 1.",
    )
    args.add_argument(
        "--distance",
        type=float,
        default=200,
        help="Distance in parsecs to search for targets. Default is 200.",
    )
    args.add_argument(
        "--identifier",
        type=str,
        default="",
        help="Identifier name for the target ecsv file - will be used in the output filename",
    )
    args.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for RA/Dec matching when replacing EU catalog entries with Gaia entries (default: 1e-3 degrees)",
    )
    args = args.parse_args()

    if args.ecsv is None or not os.path.exists(args.ecsv):
        # Set a filename for the new ecsv file, using identifier if provided
        if args.identifier:
            args.ecsv = f"{args.identifier}_targets_unified.ecsv"
        else:
            args.ecsv = "targets_unified.ecsv"

    find_more_targets(
        args.ecsv,
        args.ra,
        args.dec,
        args.radius,
        args.distance,
        args.identifier,
        args.tolerance,
    )
