from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates
from scabha.schema_utils import clickify_parameters
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib import colors
from astropy.visualization import time_support
from astropy.time import Time
import logging
from astropy.io import fits
import dask.array as da
from dask import delayed, compute
from ducc0.fft import r2c, c2r
import numpy as np
import click
import os
import re
import glob
from omegaconf import OmegaConf
from . import LOGGER, set_console_logging_level
from art import text2art


def parse_tuples(ctx, param, value: str) -> list:
    # Parse each value (which is expected to be a string) into a tuple
    parsed_tuples = []
    for val in value:
        # Assuming the input is in the format "(1,2)" without spaces
        # Strip parentheses and split by comma
        stripped_val = val[1:-1]  # Remove the parentheses
        tuple_vals = tuple(map(int, stripped_val.split(",")))
        parsed_tuples.append(tuple_vals)
    return parsed_tuples


# Function to find a valid input directory containing TARGET and TARGET_W subdirectories
def find_valid_root(input: str) -> str:
    required_dirs = {"TARGET", "TARGET_W"}
    for dirpath, dirnames, _ in os.walk(input):
        if required_dirs.issubset(set(dirnames)):
            return dirpath
    raise FileNotFoundError(
        f"No directory containing {required_dirs} found under {input}"
    )

def convert_to_list_of_tuples(input_tuple):
    return [tuple(map(float, item.split(','))) for item in input_tuple]


schemas = OmegaConf.load(os.path.join(os.path.dirname(__file__), "inspect_dynspec.yml"))


# Create entry point for the script
@click.command(help=schemas.cabs.get("inspect_dynspec").info)
@clickify_parameters(schemas.cabs.get("inspect_dynspec"))
def inspect_dynspec(
    input: str,
    output: str,
    kernel: list,
    nu_bounds: tuple,
    t_bounds: tuple,
    n_threads: int,
    stokes: str,
    std_scale: float,
    debug: bool,
    plot_for_paper: bool,
    calc_circular_pol: bool,
    calc_linear_pol: bool,
    zero_sub_value_tolerance: float,
    cmap: str,
    dpi: int,
    verbose: bool,
) -> None:

    script_name = text2art("Inspect Dynspec")
    description = "Dynamic spectra denoising and smoothing for DynSpecMS products"
    print(script_name)
    print(description)

    kernel = convert_to_list_of_tuples(kernel)

    if plot_for_paper:
        figsize = (6, 3)
    else:
        figsize = (12, 6)

    # Ensure the correct input directory is used
    input = find_valid_root(input)
    LOGGER.info(f"Found input for DynSpecMS products at {input}")

    if verbose:
        set_console_logging_level(logging.DEBUG)
        LOGGER.debug("Enabling extra verbose output")

    # kernel = parse_tuples(None, None, kernel)
    cmap = check_colormap(cmap)

    paths = get_files_paths(input)
    nof_targets = len(paths["target"]["data"])
    nof_off_targets = len(paths["off_target"]["data"])
    LOGGER.info(
        f"Found {nof_targets} target files and {nof_off_targets} off-target files."
    )
    if nof_off_targets == 0:
        LOGGER.warning(
            "No off-target files found. excess denoising not possible. Only performing analytical denoising"
        )

    # Create output directory if it does not exist
    output = os.path.join(os.path.abspath(input), os.path.abspath(output))
    if not os.path.exists(output):
        try:
            os.makedirs(output)
            LOGGER.info(f"Creating output directory {output}")
        except:
            LOGGER.error(f"Failed to create output directory {output}")
            return

    # Start major loop to iterate through targets:
    for target in range(nof_targets):

        stokes_indices = [i for i, char in enumerate("IQUV") if char in stokes]
        if stokes_indices:
            stokes_slice = np.array(stokes_indices, dtype=int)
        else:
            LOGGER.error(
                "Invalid Stokes parameters. Please provide any combination of 'I', 'Q', 'U', 'V'."
            )
            return

        nu_slice = slice(nu_bounds[0], None if nu_bounds[1] == -1 else nu_bounds[1] + 1)
        t_slice = slice(t_bounds[0], None if t_bounds[1] == -1 else t_bounds[1] + 1)

        # target_data in Jy
        target_data, target_header, target_weights, target_weights2 = (
            fetch_fits_and_weights_dask(
                paths,
                stokes_slice,
                nu_slice,
                t_slice,
                off_target=False,
                index=target,
            ).compute()
        )

        LOGGER.info(
            f"""
            Processing target {target + 1}/{nof_targets}...:\n
            Project Name: {target_header["NAME"]}\n
            Source Type: {target_header["SRC-TYPE"]}\n
            Obs ID: {target_header["OBSID"]}"""
        )
        dec_deg = np.round(np.rad2deg(target_header["DEC_RAD"]), 2)
        ra_deg = np.round(np.rad2deg(target_header["RA_RAD"]), 2)
        name_str = f"{target_header['NAME']} {target_header['SRC-TYPE']}"
        coord_str = f"$RA={ra_deg}^\degree$ and $DEC={dec_deg}^\degree$"

        if debug:
            t_weight_plot_name = os.path.join(
                output, f"{name_str.replace(' ', '_')}_{ra_deg}_{dec_deg}_W.png"
            )
            t_weight_title = f"Target weights (W) for {name_str}\nat {coord_str}"
            vminmax = (
                np.min(target_weights[target_weights != 0]),
                np.max(target_weights),
            )
            plot_dynspec(
                target_weights,
                t_weight_plot_name,
                target_header,
                nu_slice,
                t_slice,
                vminmax,
                dpi,
                cmap,
                title=t_weight_title,
                figsize=figsize,
            )
            LOGGER.info(f"Wrote W weights plot to {t_weight_plot_name}")
            t2weight_plot_name = os.path.join(
                output, f"{name_str.replace(' ', '_')}_{ra_deg}_{dec_deg}_W2.png"
            )
            t2weight_title = f"Target weights (W2) for {name_str}\nat {coord_str}"
            vminmax = (
                np.min(target_weights2[target_weights2 != 0]),
                np.max(target_weights2),
            )
            plot_dynspec(
                target_weights2,
                t2weight_plot_name,
                target_header,
                nu_slice,
                t_slice,
                vminmax,
                dpi,
                cmap,
                title=t2weight_title,
                figsize=figsize,
            )
            LOGGER.info(f"Wrote W2 weights plot to {t2weight_plot_name}")

        # As we have sliced the data already, all matrices from here on will be sliced to shape.
        """
        ################# DETERMINE DATA REGIONS ################################
        """
        mask = get_mask(target_data, blow_up_scale=1e4)
        # target_data = np.ones(target_data.shape) * 1e-3
        # target_data *= mask  # Jy
        if debug:
            mask_title = f"Flagged regions for {name_str}\nat {coord_str}"
            mask_plot_name = os.path.join(
                output,
                f"{name_str.replace(' ', '_')}_{target_header['RA_RAD']}_{target_header['DEC_RAD']}_flagged_regions.png",
            )
            vminmax = (np.min(mask), np.max(mask))
            plot_dynspec(
                data=mask,
                output=mask_plot_name,
                header=target_header,
                nu_slice=nu_slice,
                t_slice=t_slice,
                vminmax=vminmax,
                dpi=dpi,
                cmap=cmap,
                title=mask_title,
                figsize=figsize,
            )
            LOGGER.info(f"Wrote mask plot to {mask_plot_name}")

        """
        ################### ANALYTICAL DENOISING ##############################
        """
        var_a = get_analytical_variance(target_weights, target_weights2)
        target_data_a_whitened = target_data / np.sqrt(
            np.where(var_a == 0, 1, var_a)
        )  # Jy
        LOGGER.info("Completed analytical denoising")
        if debug:
            var_a_plot_name = os.path.join(
                output,
                f"{name_str.replace(' ', '_')}_{round(target_header['RA_RAD'],ndigits=2)}_{round(target_header['DEC_RAD'],ndigits=2)}_var_a.png",
            )
            var_a_title = f"Analytical variance for {name_str}\nat {coord_str}"
            vminmax = (0, std_scale * np.std(var_a))
            plot_dynspec(
                var_a,
                var_a_plot_name,
                target_header,
                nu_slice,
                t_slice,
                vminmax,
                dpi,
                cmap,
                title=var_a_title,
                figsize=figsize,
            )
            LOGGER.info(f"Wrote analytical variance plot to {var_a_plot_name}")

        """
        ################### EXCESS DENOISING #############################
        """
        if nof_off_targets != 0:
            var_e = get_excess_variance(
                target_data_a_whitened, paths, stokes_slice, nu_slice, t_slice
            )
            var = var_a * var_e
            wgt = (1 / np.where(var == 0, 1, var)) * mask
            target_data_var_normalised = target_data * wgt  # SNR
            LOGGER.info("Completed excess denoising.")
        else:
            target_data_var_normalised = target_data_a_whitened.copy()

        """
        ################### PLOT DENOISING PROGRESSION #####################################
        """
        if debug:
            for stx_idx, stx in enumerate(stokes_slice):
                stx_str = "IQUV"[stx]
                denoise_prog_name = os.path.join(
                    output,
                    f"{name_str.replace(' ', '_')}_{round(target_header['RA_RAD'],ndigits=2)}_{round(target_header['DEC_RAD'],ndigits=2)}_stokes_{stx_str}_denoise_progression.png",
                )
                denoise_title = f"{name_str} stokes {stx_str} at {coord_str} \n Left: Raw, Centre: analytically denoised, Right: excess denoised"
                plot_denoising_progression(
                    target_data[stx_idx, :, :],
                    target_data_a_whitened[stx_idx, :, :],
                    target_data_var_normalised[stx_idx, :, :],
                    nu_slice,
                    t_slice,
                    denoise_prog_name,
                    target_header,
                    std_scale,
                    dpi,
                    cmap,
                    title=denoise_title,
                    figsize=figsize,
                )
                LOGGER.info(
                    f"Wrote denoising progression plot for Stokes {stx_str} to {denoise_prog_name}"
                )

                var_e_plot_name = os.path.join(
                    output,
                    f"{name_str.replace(' ', '_')}_{round(target_header['RA_RAD'],ndigits=2)}_{round(target_header['DEC_RAD'],ndigits=2)}_stokes_{stx_str}_var_e.png",
                )
                var_e_title = (
                    f"excess variance for {name_str}\nstokes {stx_str} at {coord_str}"
                )
                vminmax = (0, std_scale * np.std(var_e[stx_idx, :, :]))
                plot_dynspec(
                    var_e[stx_idx, :, :],
                    var_e_plot_name,
                    target_header,
                    nu_slice,
                    t_slice,
                    vminmax,
                    dpi,
                    cmap,
                    title=var_e_title,
                    cbar_label="SNR",
                    figsize=figsize,
                )
                LOGGER.info(
                    f"Wrote excess variance plot for Stokes {stx_str} to {var_e_plot_name}"
                )

        """
        ################### OPTIONALLY CALCULATE CIRCULAR POLARISATION ########################
        """
        if calc_circular_pol:
            # check that both I and V stokes parameters were specified:
            if 0 not in stokes_slice or 3 not in stokes_slice:
                LOGGER.warning(
                    "To calculate circular polarisation, both Stokes I and V must be specified, skipping..."
                )
            else:
                circ_data_e_a_denoised = calc_circ_polarisation(
                    target_data_var_normalised, stokes_slice, zero_sub_value_tolerance
                )
                # add index 4 to stokes slice:
                stokes_slice = np.append(stokes_slice, 4)

                target_data_var_normalised = np.append(
                    target_data_var_normalised, circ_data_e_a_denoised, axis=0
                )
                # we must also extend wgt to account for the new stokes parameter (1's)
                wgt = np.append(wgt, np.ones(wgt[0, :, :].shape)[np.newaxis, :, :], axis=0)

        """
        ################### OPTIONALLY CALCULATING LINEAR POLARISATION ##########################
        """
        if calc_linear_pol:
            # check that I Q and U stokes parameters were specified:
            if 0 not in stokes_slice or 1 not in stokes_slice or 2 not in stokes_slice:
                LOGGER.warning(
                    "To calculate linear polarisation, Stokes I, Q and U must be specified, skipping..."
                )
            else:
                linear_data_e_a_denoised = calc_linear_polarisation(
                    target_data_var_normalised, stokes_slice, zero_sub_value_tolerance
                )
                # add index 5 to stokes slice:
                stokes_slice = np.append(stokes_slice, 5)

                target_data_var_normalised = np.append(
                    target_data_var_normalised, linear_data_e_a_denoised, axis=0
                )
                # we must also extend wgt to account for the new stokes parameter (1's)
                wgt = np.append(wgt, np.ones(wgt[0, :, :].shape)[np.newaxis, :, :], axis=0)

        stokes_slice = np.unique(stokes_slice)

        """
        ################### ITERATE THROUGH PROVIDED KERNELS #################################
        """
        for k_width in kernel:
            nu_delta, t_delta = k_width
            kern_str = f"$\Delta\\nu={np.round(nu_delta)}$MHz and $\Delta t={np.round(t_delta)}$s"

            """
            ################### SMOOTHING TO FURTHER REDUCE NOISE ##############################
            """
            # smoothed_target_data_var_normalised is sdata
            conv_target_data_var_normalised, cwgt = convolve(
                target_data_var_normalised,
                wgt,
                target_header,
                nu_delta,
                t_delta,
                nu_slice,
                t_slice,
                mask,
                n_threads,
            )
            cwgt_nonzero = np.where(cwgt == 0, 1, cwgt)
            smoothed_target_data_var_normalised = (
                conv_target_data_var_normalised / cwgt_nonzero
            )

            if debug:
                # smooth target data
                conv_target_data, _ = convolve(
                    target_data,
                    np.ones(target_data[0, :, :].shape),
                    target_header,
                    nu_delta,
                    t_delta,
                    nu_slice,
                    t_slice,
                    mask,
                    n_threads,
                )
                smoothed_target_data = conv_target_data

                # smooth target data analytically denoised
                conv_target_data_a_whitened, cwgt_white = convolve(
                    target_data_a_whitened,
                    (1 / np.sqrt(np.where(var_a == 0, 1, var_a))) * mask,
                    target_header,
                    nu_delta,
                    t_delta,
                    nu_slice,
                    t_slice,
                    mask,
                    n_threads,
                )
                smoothed_target_data_a_denoised = (
                    conv_target_data_a_whitened
                    / np.where(cwgt_white == 0, 1, cwgt_white)
                )

            for stx_idx, stx in enumerate(stokes_slice):
                stx_str = "IQUV"[stx]

                sdata_title = (
                    ""
                    if plot_for_paper
                    else f"Analytically and excess denoised, stokes {stx_str}\nfor {name_str} at {coord_str} with kernel {kern_str}"
                )
                sdata_plot_name = os.path.join(
                    output,
                    f"{name_str.replace(' ', '_')}_{round(target_header['RA_RAD'],ndigits=2)}_{round(target_header['DEC_RAD'],ndigits=2)}_stokes_{stx_str}_{int(nu_delta)}MHz_{int(t_delta)}s_data_a_e_denoise.png",
                )
                vminmax = (
                    -std_scale
                    * np.std(smoothed_target_data_var_normalised[stx_idx, :, :]),
                    std_scale
                    * np.std(smoothed_target_data_var_normalised[stx_idx, :, :]),
                )
                plot_smoothed_data(
                    smoothed_target_data_var_normalised[stx_idx, :, :],
                    nu_slice,
                    t_slice,
                    nu_delta,
                    t_delta,
                    output=sdata_plot_name,
                    header=target_header,
                    vminmax=vminmax,
                    vcenter=0,
                    dpi=dpi,
                    cmap=cmap,
                    title=sdata_title,
                    cbar_label="mJy",
                    figsize=figsize,
                    return_plot=False,
                )
                LOGGER.info(
                    f"Wrote sdata smoothed plot for stokes {stx_str} to {sdata_plot_name}"
                )

                if debug:
                    data_raw_title = (
                        ""
                        if plot_for_paper
                        else f"raw target, stokes {stx_str} for {name_str}\nat {coord_str}\nwith kernel {kern_str}"
                    )
                    data_raw_plot_name = os.path.join(
                        output,
                        f"{name_str.replace(' ', '_')}_{round(target_header['RA_RAD'],ndigits=2)}_{round(target_header['DEC_RAD'],ndigits=2)}_stokes_{stx_str}_{int(nu_delta)}MHz_{int(t_delta)}s_rawdata.png",
                    )
                    vminmax = (
                        -std_scale * np.std(smoothed_target_data[stx_idx, :, :]),
                        std_scale * np.std(smoothed_target_data[stx_idx, :, :]),
                    )
                    plot_smoothed_data(
                        smoothed_target_data[stx_idx, :, :],
                        nu_slice,
                        t_slice,
                        nu_delta,
                        t_delta,
                        output=data_raw_plot_name,
                        header=target_header,
                        vminmax=vminmax,
                        vcenter=0,
                        dpi=dpi,
                        cmap=cmap,
                        title=data_raw_title,
                        cbar_label="mJy",
                        figsize=figsize,
                        return_plot=False,
                    )
                    LOGGER.info(
                        f"Wrote target smoothed plot for stokes {stx_str} to {data_raw_plot_name}"
                    )

                    data_a_title = (
                        ""
                        if plot_for_paper
                        else f"Analytically denoised target, stokes {stx_str} for {name_str}\nat {coord_str}\nwith kernel {kern_str}"
                    )
                    data_a_plot_name = os.path.join(
                        output,
                        f"{name_str.replace(' ', '_')}_{round(target_header['RA_RAD'],ndigits=2)}_{round(target_header['DEC_RAD'],ndigits=2)}_stokes_{stx_str}_{int(nu_delta)}MHz_{int(t_delta)}s_data_a_denoise.png",
                    )
                    vminmax = (
                        -std_scale
                        * np.std(smoothed_target_data_a_denoised[stx_idx, :, :]),
                        std_scale
                        * np.std(smoothed_target_data_a_denoised[stx_idx, :, :]),
                    )
                    plot_smoothed_data(
                        smoothed_target_data_a_denoised[stx_idx, :, :],
                        nu_slice,
                        t_slice,
                        nu_delta,
                        t_delta,
                        output=data_a_plot_name,
                        header=target_header,
                        vminmax=vminmax,
                        vcenter=0,
                        dpi=dpi,
                        cmap=cmap,
                        title=data_a_title,
                        cbar_label="mJy",
                        figsize=figsize,
                        return_plot=False,
                    )
                    LOGGER.info(
                        f"Wrote target smoothed plot for stokes {stx_str} to {data_a_plot_name}"
                    )


@delayed
def fetch_fits_and_weights_dask(
    paths, stokes_slice, nu_slice, t_slice, off_target, index
):
    target_str = "off_target" if off_target else "target"

    with fits.open(paths[target_str]["data"][index]) as hdu_list:
        target_data = (
            hdu_list[0].data[stokes_slice, nu_slice, t_slice].astype(np.float64)
        )
        target_header = hdu_list[0].header

    with fits.open(paths[target_str]["target_W"][index]) as hdu_list_weights:
        target_weights = hdu_list_weights[0].data[0, nu_slice, t_slice]

    with fits.open(paths[target_str]["target_W2"][index]) as hdu_list_weights2:
        target_weights2 = hdu_list_weights2[0].data[0, nu_slice, t_slice]

    return target_data, target_header, target_weights, target_weights2


def check_colormap(colormap_name):
    valid_colormaps = plt.colormaps()
    if colormap_name not in valid_colormaps:
        LOGGER.warning(
            f"Provided colormap {colormap_name} is not valid. Using default colormap 'inferno'..."
        )
        colormap_name = "inferno"
    return colormap_name


def plot_dynspec(
    data: np.ndarray,
    output: str,
    header: fits.header.Header,
    nu_slice: slice,
    t_slice: slice,
    vminmax: tuple,
    dpi: int = 300,
    cmap: str = "inferno",
    title: str = "",
    cbar_label: str = "",
    return_plot: bool = False,
    figsize: tuple = (12, 6),
) -> Optional[Axes]:
    """
    Given the data data, plot the data.
    Args:
        data: 2D array of data.
        output: Filename for the plot.
        header: FITS header.
        nu_slice: Slice of the frequency range to consider.
        t_slice: Slice of the time range to consider.
        vminmax: Plot saturates at vmin,vmax = vminmax in plots
        dpi: DPI of the output plots
        cmap: Colormap to use for plotting.
        title: Title to add to the plot.
        cbar_label: Label for the colourbar.
        return_plot: Whether to return the Axes object for external plotting.
        figsize: Size of the figure.
    Returns:
        Optionally returns Axes object for external plotting.
    """
    t_ticks = fetch_t_ticks_mjd(header)[t_slice]
    t0, t1 = t_ticks[0].to_datetime(), t_ticks[-1].to_datetime()
    nu_ticks = fetch_axis_ticks(header, axis=2)[nu_slice]
    vmin, vmax = vminmax

    _, ax = plt.subplots(1, 1, figsize=figsize)
    with time_support(simplify=True):
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=[t0, t1, nu_ticks[0], nu_ticks[-1]],
            origin="lower",
            interpolation="none",
        )
        # Add the colorbar
        if cbar_label == "" or cbar_label == "SNR":
            cbar = plt.colorbar(im, ax=ax)
        else:
            cbar = plt.colorbar(im, ax=ax, format=FuncFormatter(format_func))
            cbar.set_label(cbar_label)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=6)
        )  # Control the number of y-ticks
        ax.yaxis.set_major_formatter(FuncFormatter(format_func_ghz))
        ax.set_xlim(t0, t1)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Frequency, GHz")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_title(title)
        output = output.replace(":", "_").replace("-", "")
    plt.savefig(output, dpi=dpi)
    if return_plot:
        return ax
    plt.close()
    return None


def get_mask(data: np.ndarray, blow_up_scale: float = 1e4) -> np.ndarray:
    """
    Blows data values up by blow_up_scale and flags regions equal to exactly 0. Does not return
    per polarization as it assumes flagged regions are identical across polarizations.
    Args:
        data: 3D array of data (n_pol, n_freq, n_time).
    Returns:
        2D array of flagged regions.
    """
    _, n_freq, n_time = data.shape
    mask = np.ones((n_freq, n_time))
    data = data * blow_up_scale
    mask = np.where(data[0, :, :] == 0, 0, mask)
    return mask


def calc_circ_polarisation(
    arg_data: np.ndarray, stokes_slice: np.ndarray, tolerance: float = 1e-4
) -> np.ndarray:
    """
    Args:
        data: 3D array of data (n_pol, n_freq, n_time).
        stokes_slice: Slice of the Stokes parameters to consider.
    Returns:
        3D array of circular polarisation fraction V/I.
    """
    data = arg_data.copy()
    v_indices = np.where(stokes_slice == 3)[0][0]
    i_indices = np.where(stokes_slice == 0)[0][0]

    I = data[i_indices, :, :]
    V = data[v_indices, :, :]
    I = np.where(np.isclose(I, 0, atol=tolerance), tolerance, I)
    circ_pol = V[np.newaxis, :, :] / I
    return circ_pol


def calc_linear_polarisation(
    arg_data: np.ndarray, stokes_slice: np.ndarray, tolerance: float = 1e-4
) -> np.ndarray:
    """
    Args:
        data: 3D array of data (n_pol, n_freq, n_time).
        stokes_slice: Slice of the Stokes parameters to consider.
    Returns:
        3D array of linear polarisation fraction sqrt((Q^2 + U^2) / I^2).
    """
    data = arg_data.copy()
    q_indices = np.where(stokes_slice == 1)[0][0]
    u_indices = np.where(stokes_slice == 2)[0][0]
    i_indices = np.where(stokes_slice == 0)[0][0]
    I = data[i_indices, :, :]
    Q = data[q_indices, :, :]
    U = data[u_indices, :, :]
    I = np.where(np.isclose(I, 0, atol=tolerance), tolerance, I)
    linear_pol = np.sqrt((Q**2 + U**2) / I**2)
    return linear_pol[np.newaxis, :, :]


def fetch_t_ticks_mjd(header: fits.header.Header) -> Time:
    """
    Args:
        header: FITS header.
    Returns:
        List of time ticks in MJD.
    """
    # Get the start and stop times from the header
    obs_start = Time(header["OBS-STAR"], format="isot", scale="utc")
    obs_stop = Time(header["OBS-STOP"], format="isot", scale="utc")

    # Calculate the total duration of the observation in seconds
    total_duration = (obs_stop - obs_start).sec

    # Get the time ticks from the header
    t_ticks = fetch_axis_ticks(header, axis=1)

    # Convert the time ticks to MJD
    time_ticks_mjd = obs_start.mjd + (t_ticks / total_duration) * (
        obs_stop.mjd - obs_start.mjd
    )

    return Time(time_ticks_mjd, format="mjd")


def fetch_axis_ticks(header: fits.header.Header, axis: int) -> np.ndarray:
    """
    Args:
        header: FITS header.
        axis: Axis to fetch the ticks for.
    Returns:
        List of ticks for the axis.
    """
    npix = header["NAXIS" + str(axis)]
    refpix = header["CRPIX" + str(axis)] - 1  # zero indexing
    delta = header["CDELT" + str(axis)]
    ref_val = header["CRVAL" + str(axis)]
    return ref_val + np.arange(refpix, npix) * delta


# Function to format colorbar values
def format_func(value, tick_number):
    return f"{np.round(value * 1e3, decimals=2)}"


# Function to format smooth line plot
def format_func_line(value, tick_number):
    return f"{np.round(value,decimals=2)}"


# Function to format frequency values - assumes input is in MHz
def format_func_ghz(value, tick_number):
    return f"{np.round(value * 1e-3, decimals=2)}"


def plot_smoothed_data(
    smoothed_data: np.ndarray,
    nu_slice: slice,
    t_slice: slice,
    nu_delta: float,
    t_delta: float,
    output: str,
    header: fits.header.Header,
    vminmax: tuple,
    vcenter: float,
    dpi: int,
    cmap: str,
    title: str = "",
    cbar_label: str = "",
    figsize: tuple = (12, 6),
    return_plot: bool = False,
) -> Optional[Axes]:
    """
    Args:
        smoothed_data: 2D array of smoothed data.
        nu_slice: Slice of the frequency range to consider.
        t_slice: Slice of the time range to consider.
        nu_delta: FWHM frequency width of the kernel.
        t_delta: FWHM time width of the kernel.
        output: Plot filename.
        header: FITS header.
        vminmax: Plot saturates at vmin,vmax = vminmax in plots
        vcenter: Center of the colorbar.
        dpi: DPI of the output plots
        cmap: Colormap to use for plotting
        title: Title to add to the plot.
        cbar_label: Label for the color
        figsize: Size of the figure.
        return_plot: Whether to return the Axes object for external plotting.
    Returns:
        Optionally returns Axes object for external plotting.
    """
    t_ticks = fetch_t_ticks_mjd(header)[t_slice]
    t0, t1 = t_ticks[0].to_datetime(), t_ticks[-1].to_datetime()
    nu_ticks = fetch_axis_ticks(header, axis=2)[nu_slice]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    vmin, vmax = vminmax
    plt.subplots_adjust(hspace=0.1)  # Adjust the space between subplots
    # Top plot
    with time_support(simplify=True):
        colornorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im0 = ax.imshow(
            smoothed_data,
            cmap=cmap,
            aspect="auto",
            interpolation="none",
            norm=colornorm,
            extent=[t0, t1, nu_ticks[0], nu_ticks[-1]],
            origin="lower",
        )
        ax.set_ylabel("$\\nu$ (GHz)")
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=6)
        )  # Control the number of y-ticks
        ax.yaxis.set_major_formatter(FuncFormatter(format_func_ghz))
        # Get the position of the top plot
        pos = ax.get_position()
        cax0_bottom = pos.y0
        cax0_height = pos.height
        cax0 = fig.add_axes([pos.x1 + 0.02, cax0_bottom, 0.015, cax0_height])
        cbar0 = fig.colorbar(
            im0, cax=cax0, spacing="proportional", orientation="vertical"
        )
        cbar0.ax.set_yscale("linear")
        if cbar_label == "mJy":
            cbar0.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        cbar0.set_label(cbar_label)

        # Define the margin as a fraction of the total ranges
        total_time_range_seconds = (t1 - t0).total_seconds()
        twin_y_axis = ax.twiny()
        twin_y_axis.set_xlim(0, len(t_ticks) - 1)
        # Hide the y-ticks and y-labels for the twin axis
        twin_y_axis.xaxis.set_ticks([])
        twin_y_axis.xaxis.set_ticklabels([])

        ellipse_center_x = len(t_ticks) - 125
        ellipse_center_y = nu_ticks[-1] - 55
        ellipse_width_time = t_delta / total_time_range_seconds * len(t_ticks)

        # Create the ellipse
        kernel_ellipse = patches.Ellipse(
            (ellipse_center_x, ellipse_center_y),
            width=ellipse_width_time,
            height=nu_delta,
            edgecolor="black",
            facecolor="gainsboro",
            linewidth=1,
        )
        twin_y_axis.add_patch(kernel_ellipse)

        ax.set_xlim(t0, t1)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.set_xlabel("Time (UTC)")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    plt.suptitle(title)
    output = output.replace(":", "_").replace("-", "")
    plt.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
    if return_plot:
        return ax
    plt.close()
    return None


def plot_denoising_progression(
    target_data: np.ndarray,
    target_a_denoised: np.ndarray,
    target_a_e_denoised: np.ndarray,
    nu_slice: slice,
    t_slice: slice,
    output: str,
    header: fits.header.Header,
    std_scale: float,
    dpi: int,
    cmap: str,
    title: str = "",
    figsize: tuple = (12, 6),
    return_plot: bool = False,
) -> Optional[Axes]:
    """
    Plot a progression from left to right of the denoising process.
    Args:
        target_data: 2D array of data (n_freq, n_time)
        target_a_denoised: 3D array of analytically denoised data
        target_a_e_denoised: 3D array of excessly denoised data
        nu_slice: Slice of the frequency range to consider
        t_slice: Slice of the time range to consider
        output: Output file name
        header: FITS header
        std_scale: Plot saturates at std_scale * std(data) in plots
        dpi: DPI of the output plots
        cmap: Colormap to use for plotting
        title: Title of the plot
        return_plot: Whether to return the Axes object for external plotting
    Returns:
        Optionally returns Axes object for external plotting
    """
    t_ticks = fetch_t_ticks_mjd(header)[t_slice]
    t0, t1 = t_ticks[0].to_datetime(), t_ticks[-1].to_datetime()
    nu_ticks = fetch_axis_ticks(header, axis=2)[nu_slice]

    aspect_ratio_t = (
        target_data.shape[1] / target_data.shape[0]
        if target_data.shape[0] > target_data.shape[1]
        else 1
    )
    aspect_ratio_f = (
        target_data.shape[0] / target_data.shape[1]
        if target_data.shape[1] > target_data.shape[0]
        else 1
    )

    _, ax = plt.subplots(
        1,
        3,
        figsize=(
            (figsize[0] * aspect_ratio_t) + 1.6,
            (figsize[1] * aspect_ratio_f) + 3.3,
        ),
        sharex=True,
        sharey=True,
    )
    with time_support(simplify=True):
        im0 = ax[0].imshow(
            target_data,
            cmap=cmap,
            vmin=-std_scale * np.std(target_data),
            vmax=std_scale * np.std(target_data),
            aspect="auto",
            extent=[t0, t1, nu_ticks[0], nu_ticks[-1]],
            interpolation="none",
            origin="lower",
        )
        ax[0].set_xlabel("Time (UTC)")
        ax[0].set_ylabel("Frequency, GHz")
        cbar = plt.colorbar(im0, ax=ax[0], format=FuncFormatter(format_func))
        cbar.set_label("mJy")
        ax[0].yaxis.set_major_locator(
            MaxNLocator(nbins=6)
        )  # Control the number of y-ticks
        ax[0].yaxis.set_major_formatter(FuncFormatter(format_func_ghz))
        ax[0].set_xlim(t0, t1)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax[0].xaxis.set_major_locator(locator)
        ax[0].xaxis.set_major_formatter(formatter)
        output = output.replace(":", "_").replace("-", "")

        im1 = ax[1].imshow(
            target_a_denoised,
            cmap=cmap,
            vmin=-std_scale * np.std(target_a_denoised),
            vmax=std_scale * np.std(target_a_denoised),
            aspect="auto",
            extent=[t0, t1, nu_ticks[0], nu_ticks[-1]],
            interpolation="none",
            origin="lower",
        )
        ax[1].set_xlabel("Time (UTC)")
        ax[1].set_ylabel("Frequency, GHz")
        cbar = plt.colorbar(im1, ax=ax[1], format=FuncFormatter(format_func))
        cbar.set_label("mJy")
        ax[1].yaxis.set_major_locator(
            MaxNLocator(nbins=6)
        )  # Control the number of y-ticks
        ax[1].yaxis.set_major_formatter(FuncFormatter(format_func_ghz))
        ax[1].set_xlim(t0, t1)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax[1].xaxis.set_major_locator(locator)
        ax[1].xaxis.set_major_formatter(formatter)
        output = output.replace(":", "_").replace("-", "")

        im2 = ax[2].imshow(
            target_a_e_denoised,
            cmap=cmap,
            vmin=-std_scale * np.std(target_a_e_denoised),
            vmax=std_scale * np.std(target_a_e_denoised),
            aspect="auto",
            extent=[t0, t1, nu_ticks[0], nu_ticks[-1]],
            interpolation="none",
            origin="lower",
        )
        ax[2].set_xlabel("Time (UTC)")
        ax[2].set_ylabel("Frequency, GHz")
        cbar = plt.colorbar(im2, ax=ax[2])
        cbar.set_label("SNR")
        ax[2].yaxis.set_major_locator(
            MaxNLocator(nbins=6)
        )  # Control the number of y-ticks
        ax[2].yaxis.set_major_formatter(FuncFormatter(format_func_ghz))
        ax[2].set_xlim(t0, t1)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax[2].xaxis.set_major_locator(locator)
        ax[2].xaxis.set_major_formatter(formatter)
        output = output.replace(":", "_").replace("-", "")

    plt.suptitle(title)

    plt.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
    if return_plot:
        return ax
    plt.close()


def fold_dynspec(
    data: np.ndarray, mask: np.ndarray, weights=None, data_axis=1, weight_axis=1
) -> np.ndarray:
    """
    Args:
        data: 3D array of data (n_pol, n_freq, n_time).
        mask: 2D array of flagged regions.
        weights: 3D array of weights that if provided are used for normalisation.
        normalise: Whether to normalise the data after folding.
        data_axis: Data axis to fold the data along.
        weight_axis: Weight axis to fold the weights along.
    Returns:
        array of folded data reduced by dimension in which folded
        and where weights are provided, the propagated error of the folded data
        calculated as sqrt(variance * weight / sum(weight))
    """
    if weights is not None:
        norm = np.sum(weights, axis=weight_axis)  # non-uniform weights
        norm = np.where(norm == 0, 1, norm)  # avoid division by zero
    else:
        norm = np.count_nonzero(data, axis=data_axis)  # uniform weights
        norm = np.where(norm == 0, 1, norm)  # avoid division by zero

    mask_norm = np.count_nonzero(mask, axis=0)
    sum_mask = np.sum(mask, axis=0) / np.where(mask_norm == 0, 1, mask_norm)
    if not np.all(np.logical_or(sum_mask == 0, sum_mask == 1)):
        raise ValueError("Mask norm should only contain values 0 or 1")

    std_dev = (np.sqrt(1 / norm)) * sum_mask

    folded_data = (np.sum(data, axis=data_axis)) / norm
    return folded_data, std_dev


def make_kernel(x, l):
    """
    Args:
        x: 1D array of data.
        l: FWHM of the kernel.
    Returns:
        1D array of the kernel.
    """
    l_sigma = l / 2 * np.sqrt(2 * np.log(2))
    nx = x.size
    return np.exp(-((x - x[nx // 2]) ** 2) / (2 * l_sigma**2))


def convolve(
    data: np.ndarray,
    weight: Optional[np.ndarray],
    header: fits.header.Header,
    nu_delta: float,
    t_delta: float,
    nu_slice: slice,
    t_slice: slice,
    mask: np.ndarray,
    n_threads: int = 32,
) -> np.ndarray:
    """
    Args:
        data: 3D array of data (n_pol, n_freq, n_time).
        weight: 3D array of weights (n_pol, n_freq, n_time). If None, no weight is applied
        header: FITS header.
        nu_delta: Frequency delta in MHz specifying FWHM of the smoothing kernel.
        t_delta: Time delta in seconds specifying FWHM of the smoothing kernel.
        nu_slice: Slice of the frequency range to consider.
        t_slice: Slice of the time range to consider.
        mask: 3D array of flagged regions. Divide by the convolved mask if not weight is supplied
        n_threads: Number of threads to use for FFT operations.

    Returns:
        3D array of cdata.
        3D array of cwgt if weights are provided - else None.
    """
    _, _, t_axis = data.shape

    phys_time, delta_time = get_refval_and_axisvals(header, axis=1)
    phys_time -= phys_time.min()
    phys_time /= 3600  # sec to hr
    phys_freq, delta_freq = get_refval_and_axisvals(header, axis=2)
    phys_time = phys_time[t_slice]
    phys_freq = phys_freq[nu_slice]

    t_delta = t_delta if t_delta >= delta_time else delta_time
    nu_delta = nu_delta if nu_delta >= delta_freq else delta_freq

    # normalize range and kernel
    nu = phys_freq - phys_freq[0]
    lnu = nu_delta / nu.max()
    nu /= nu.max()
    t = phys_time - phys_time[0]
    lt = t_delta / t.max() / 3600
    t /= t.max()

    # Determine the padding size - same size as kernel
    pad_t = int(t_delta)
    pad_nu = int(nu_delta)

    kt = make_kernel(t, lt)
    kv = make_kernel(nu, lnu)
    K = kv[:, None] * kt[None, :]
    K /= K.sum()
    padded_kernel = np.pad(K, ((pad_nu, pad_nu), (pad_t, pad_t)), mode="constant")

    iFs = np.fft.ifftshift
    Fs = np.fft.fftshift

    # Real-> Complex of IFFT Shift of Kernel
    Khat = r2c(
        iFs(padded_kernel), axes=(0, 1), nthreads=n_threads, forward=True, inorm=0
    )

    # Real->Complex of IFFT Shift of the data
    padded_data = np.pad(
        data, ((0, 0), (pad_nu, pad_nu), (pad_t, pad_t)), mode="constant"
    )
    datahat = r2c(
        iFs(padded_data),
        axes=(1, 2),
        nthreads=n_threads,
        forward=True,
        inorm=0,
    )
    datahat *= Khat

    # Complex->Real FFT to get cdata = (FFT(norm) x FFT(K/K.sum))
    cdata = Fs(
        c2r(
            datahat,
            axes=(1, 2),
            nthreads=n_threads,
            forward=False,
            inorm=2,
            lastsize=t_axis + 2 * pad_t,
        )
    )
    cdata = cdata[:, pad_nu:-pad_nu, pad_t:-pad_t]

    if weight.ndim == 2:
        weight = np.expand_dims(weight, axis=0)

    padded_weight = np.pad(
        weight, ((0, 0), (pad_nu, pad_nu), (pad_t, pad_t)), mode="constant"
    )
    wgthat = r2c(
        iFs(padded_weight),
        axes=(1, 2),
        nthreads=n_threads,
        forward=True,
        inorm=0,
    )

    wgthat *= Khat
    cwgt = Fs(
        c2r(
            wgthat,
            axes=(1, 2),
            nthreads=n_threads,
            forward=False,
            inorm=2,
            lastsize=t_axis + 2 * pad_t,
        )
    )
    cwgt = cwgt[:, pad_nu:-pad_nu, pad_t:-pad_t]

    # zero regions where mask is zero
    cdata *= mask
    cwgt *= mask

    return cdata, cwgt


@delayed
def process_off_target(paths, stokes_slice, nu_slice, t_slice, noff):
    off_data, _, off_data_w, off_data_w2 = fetch_fits_and_weights_dask(
        paths,
        stokes_slice,
        nu_slice,
        t_slice,
        off_target=True,
        index=noff,
    ).compute()
    off_var_a = get_analytical_variance(off_data_w, off_data_w2)
    whitened_data = off_data / np.sqrt(np.where(off_var_a == 0, 1, off_var_a))
    return whitened_data


def get_excess_variance(
    target_data_a_whitened: np.ndarray,
    paths: dict,
    stokes_slice: np.ndarray,
    nu_slice: slice,
    t_slice: slice,
) -> np.ndarray:
    nof_off_targets = len(paths["off_target"]["data"])
    init_shape = (nof_off_targets + 1,) + target_data_a_whitened.shape
    onoff_a_whitened = np.zeros(init_shape)

    # Prepare delayed tasks for Dask
    tasks = [
        process_off_target(paths, stokes_slice, nu_slice, t_slice, noff)
        for noff in range(nof_off_targets)
    ]

    # Compute the tasks in parallel
    results = compute(*tasks)

    # Collect results
    for noff, whitened_data in enumerate(results):
        onoff_a_whitened[noff, :, :, :] = whitened_data

    # Add the target data
    onoff_a_whitened[-1, :, :, :] = target_data_a_whitened

    # Convert to Dask array for parallel computation of MAD
    onoff_a_whitened_dask = da.from_array(onoff_a_whitened, chunks=(1, 1, 1024, 4163))
    onoff_a_whitened_mad = da.median(
        da.abs(onoff_a_whitened_dask - da.median(onoff_a_whitened_dask, axis=0)), axis=0
    ).compute()
    # Normalize the MAD to be equivalent to scipy.stats.median_abs_deviation with scale='normal'
    onoff_a_whitened_mad /= 0.6745
    var_e = np.power(onoff_a_whitened_mad, 2)
    return var_e


def get_analytical_variance(weights: np.ndarray, weights2: np.ndarray) -> np.ndarray:
    """
    Provided with the weights and weights^2, calculates the analytical variance as:
    var_a = W2/W**2
    Args:
        weights: 2D array of weights (n_freq, n_time)
        weights2: 2D array of weights^2 (n_freq, n_time)
    Returns:
        3D array of first order normalized data
    """
    weights = np.where(weights == 0, 1, weights)
    var_a = weights2 / np.power(weights, 2)
    return var_a


def get_refval_and_axisvals(hdr: fits.header.Header, axis: int = 3) -> tuple:
    """
    Args:
        hdr: FITS header.
        axis: Axis of the data to extract.
    Returns:
        Tuple of the data and the delta value.
    """
    npix = hdr["NAXIS" + str(axis)]
    refpix = hdr["CRPIX" + str(axis)]
    delta = hdr["CDELT" + str(axis)]
    ref_val = hdr["CRVAL" + str(axis)]
    return ref_val + np.arange(1 - refpix, npix) * delta, delta


def extract_identifier(filename):
    match = re.search(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}_-?\d{2}:\d{2}:\d{2}\.\d{3})", filename
    )
    return match.group(1) if match else None


def get_files_paths(input: str) -> dict:
    """
    Args:
        input: Path to the input directory of the data.
    Returns:
        Dictionary of paths to all targets and their respective weights.
        Structure   {
                    target:     {data: [path_1, path_2, ...],
                                target_W: [pathw1_1, pathw2_2, ...],
                                target_W2: [pathw2_1, pathw2_2, ...]},
                    off_target: {data: [path_1, path_2, ...],
                                target_W: [pathw1_1, pathw2_2, ...],
                                target_W2: [pathw2_1, pathw2_2, ...]}
                    }
    """
    paths = {}

    def group_files_by_identifier(file_list):
        grouped_files = {}
        for file in file_list:
            identifier = extract_identifier(file)
            if identifier:
                if identifier not in grouped_files:
                    grouped_files[identifier] = []
                grouped_files[identifier].append(file)
        return grouped_files

    target_data = glob.glob(os.path.join(input, "TARGET", "*.fits"))
    target_w = glob.glob(os.path.join(input, "TARGET_W", "*W.fits"))
    target_w2 = glob.glob(os.path.join(input, "TARGET_W", "*W2.fits"))

    off_data = glob.glob(os.path.join(input, "OFF", "*.fits"))
    off_w = glob.glob(os.path.join(input, "OFF_W", "*W.fits"))
    off_w2 = glob.glob(os.path.join(input, "OFF_W", "*W2.fits"))

    paths["target"] = {
        "data": sorted(target_data),
        "target_W": sorted(target_w),
        "target_W2": sorted(target_w2),
    }

    off_data_grouped = group_files_by_identifier(off_data)
    off_w_grouped = group_files_by_identifier(off_w)
    off_w2_grouped = group_files_by_identifier(off_w2)

    off_data_sorted = []
    off_w_sorted = []
    off_w2_sorted = []

    for identifier in sorted(off_data_grouped.keys()):
        off_data_sorted.extend(off_data_grouped[identifier])
        off_w_sorted.extend(off_w_grouped.get(identifier, []))
        off_w2_sorted.extend(off_w2_grouped.get(identifier, []))

    paths["off_target"] = {
        "data": off_data_sorted,
        "target_W": off_w_sorted,
        "target_W2": off_w2_sorted,
    }

    return paths
