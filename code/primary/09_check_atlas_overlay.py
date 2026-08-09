#!/usr/bin/env python3

# What this script does:
#   Makes atlas-on-anatomy overlay figures so I can visually confirm that the
#   Schaefer/Yeo atlas sits sensibly on the normalized data.
# How to run it:
#   Run from the repo root with:
#   python code/primary/09_check_atlas_overlay.py
# Main outputs:
#   data/processed/qc/atlas_overlay/figures/
#   data/processed/qc/atlas_overlay/atlas_overlay_summary.tsv

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import datasets, image, plotting


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dissertation_figure_style import (
    AXIS_TEXT_COLOR,
    TEXT_COLOR,
    TITLE_SIZE,
    TICK_SIZE,
    apply_dissertation_rcparams,
)

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#9C755F",
    "#E15759",
    "#B07AA1",
    "#F28E2B",
    "#EDC948",
]

apply_dissertation_rcparams(title_size=TITLE_SIZE, tick_size=TICK_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the Schaefer 200 / Yeo 7 atlas on each subject's preprocessed "
            "MNI-space forward resting-state BOLD image and save QC figures."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Optional subject IDs to inspect. Defaults to all subjects in --sample.",
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Sample TSV used to choose subjects when no subject IDs are given.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="Path to the fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/qc/atlas_overlay",
        help="Directory for atlas-overlay QC figures and summary TSV.",
    )
    parser.add_argument(
        "--atlas-data-dir",
        default="data/external/nilearn_data",
        help="Writable directory where Nilearn atlas files should be cached.",
    )
    parser.add_argument(
        "--n-rois",
        type=int,
        default=200,
        help="Number of Schaefer parcels to use.",
    )
    parser.add_argument(
        "--yeo-networks",
        type=int,
        default=7,
        help="Number of Yeo canonical networks to use.",
    )
    parser.add_argument(
        "--resolution-mm",
        type=int,
        default=2,
        help="Atlas resolution in millimeters.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Atlas overlay transparency for the saved figure.",
    )
    parser.add_argument(
        "--ortho-cut-coords",
        nargs=3,
        type=float,
        default=(0.0, -20.0, 20.0),
        metavar=("X", "Y", "Z"),
        help="Cut coordinates for the single orthographic overview figure.",
    )
    parser.add_argument(
        "--n-sagittal-cuts",
        type=int,
        default=8,
        help="Number of sagittal slices to save for the multi-slice sagittal view.",
    )
    parser.add_argument(
        "--n-coronal-cuts",
        type=int,
        default=8,
        help="Number of coronal slices to save for the multi-slice coronal view.",
    )
    parser.add_argument(
        "--n-axial-cuts",
        type=int,
        default=8,
        help="Number of axial slices to save for the multi-slice axial view.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display figures interactively instead of only saving them.",
    )
    parser.add_argument(
        "--save-ortho-figure",
        type=int,
        choices=(0, 1),
        default=0,
        help="Save a single orthographic parcel-overlay figure on the mean BOLD image.",
    )
    parser.add_argument(
        "--save-multislice-figures",
        type=int,
        choices=(0, 1),
        default=0,
        help="Save the multi-slice sagittal, coronal, and axial parcel-overlay figures.",
    )
    parser.add_argument(
        "--save-clean-network-figure",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Also save a cleaner dissertation-style orthographic figure using the "
            "7-network version of the atlas instead of all 200 parcels."
        ),
    )
    parser.add_argument(
        "--save-dissertation-figure",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Save the main 3-panel QC figure using the 7-network atlas on the "
            "subject's normalized T1w image."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sample(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, sep="\t")
    if "subject_id" not in sample.columns:
        raise ValueError(f"Sample TSV is missing required column: subject_id ({path})")
    return sample


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_forward_bold_path(derivatives_dir: Path, subject: str) -> Path:
    func_dir = derivatives_dir / subject / "ses-01" / "func"
    return next(
        func_dir.glob(
            f"{subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        ),
        None,
    )


def get_mni_t1w_path(derivatives_dir: Path, subject: str) -> Path:
    anat_dir = derivatives_dir / subject / "ses-01" / "anat"
    matches = sorted(
        anat_dir.glob(
            f"{subject}_ses-01*_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"
        )
    )
    return matches[0] if matches else None


def decode_label(label: object) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def parse_network_name(label: str) -> str:
    parts = label.split("_")
    if parts and parts[0].endswith("Networks") and len(parts) > 2:
        return parts[2]
    if len(parts) > 1:
        return parts[1]
    return label


def build_network_level_atlas(atlas_img: str, atlas_labels: list[str]) -> nib.Nifti1Image:
    # For visual QC I collapse parcels up to the 7 network labels so the overlay is easier to read.
    atlas_nii = nib.load(atlas_img)
    atlas_data = atlas_nii.get_fdata()
    network_data = np.zeros(atlas_data.shape, dtype=np.int16)

    parcel_to_network = {}
    for parcel_idx, label in enumerate(atlas_labels, start=1):
        network_name = parse_network_name(label)
        if network_name not in NETWORK_ORDER:
            continue
        parcel_to_network[parcel_idx] = NETWORK_ORDER.index(network_name) + 1

    for parcel_idx, network_idx in parcel_to_network.items():
        network_data[atlas_data == parcel_idx] = network_idx

    return nib.Nifti1Image(network_data, atlas_nii.affine, atlas_nii.header)


def save_overlay_figure(
    atlas_img: str,
    subject: str,
    bold_path: Path,
    output_path: Path,
    display_mode: str,
    cut_coords: list[float] | tuple[float, ...],
    alpha: float,
    show: bool,
) -> None:
    mean_bold = image.mean_img(str(bold_path))
    display = plotting.plot_roi(
        atlas_img,
        bg_img=mean_bold,
        display_mode=display_mode,
        cut_coords=cut_coords,
        title=f"{subject}: Schaefer 200 / Yeo 7 atlas overlay ({display_mode})",
        alpha=alpha,
        draw_cross=False,
        black_bg=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(str(output_path), dpi=200)
    if show:
        plotting.show()
    display.close()
    plt.close("all")


def save_clean_network_figure(
    network_atlas_img: nib.Nifti1Image,
    subject: str,
    bold_path: Path,
    output_path: Path,
    cut_coords: list[float] | tuple[float, ...],
    alpha: float,
    show: bool,
) -> None:
    mean_bold = image.mean_img(str(bold_path))
    cmap = ListedColormap(NETWORK_COLORS)
    display = plotting.plot_roi(
        network_atlas_img,
        bg_img=mean_bold,
        display_mode="ortho",
        cut_coords=cut_coords,
        title=None,
        alpha=min(alpha + 0.1, 0.6),
        draw_cross=False,
        annotate=False,
        colorbar=False,
        black_bg=False,
        cmap=cmap,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(str(output_path), dpi=300)
    if show:
        plotting.show()
    display.close()
    plt.close("all")


def save_dissertation_network_figure(
    network_atlas_img: nib.Nifti1Image,
    subject: str,
    bg_img: str | nib.Nifti1Image,
    output_path: Path,
    cut_coords: list[float] | tuple[float, ...],
    alpha: float,
) -> None:
    # This is the cleaner figure version I would actually keep for the dissertation.
    cmap = ListedColormap(NETWORK_COLORS)
    panel_titles = ["Sagittal", "Coronal", "Axial"]
    display_modes = ["x", "y", "z"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), facecolor="white")

    for ax, panel_title, display_mode, coord in zip(
        axes, panel_titles, display_modes, cut_coords
    ):
        display = plotting.plot_roi(
            network_atlas_img,
            bg_img=bg_img,
            display_mode=display_mode,
            cut_coords=[coord],
            axes=ax,
            figure=fig,
            alpha=min(alpha + 0.15, 0.65),
            draw_cross=False,
            annotate=False,
            colorbar=False,
            black_bg=False,
            cmap=cmap,
        )
        ax.set_title(panel_title, fontsize=11, pad=10, color=AXIS_TEXT_COLOR, fontweight="bold")
        display.close()

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in zip(NETWORK_ORDER, NETWORK_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
        labelcolor=TEXT_COLOR,
    )
    fig.suptitle(
        f"{subject}: Schaefer 200 / Yeo 7 atlas on normalized T1w",
        fontsize=14,
        fontweight="bold",
        color=AXIS_TEXT_COLOR,
        y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.18, wspace=0.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evenly_spaced_coords(start: float, stop: float, n_cuts: int) -> list[float]:
    if n_cuts < 1:
        raise ValueError("Number of cuts must be at least 1.")
    return np.linspace(start, stop, n_cuts).round(1).tolist()


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    output_dir = resolve_project_path(args.output_dir)
    atlas_data_dir = resolve_project_path(args.atlas_data_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    atlas_data_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(sample_path)
    if args.subjects:
        requested = set(args.subjects)
        sample = sample[sample["subject_id"].isin(requested)].copy()
        missing = requested.difference(set(sample["subject_id"]))
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Requested subjects not found in sample TSV: {missing_str}")

    if sample.empty:
        raise ValueError("No subjects remain to inspect.")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
        data_dir=str(atlas_data_dir),
    )
    # I fetch the exact atlas version used later in extraction so the visual check matches the analysis.
    atlas_labels = [decode_label(label) for label in atlas.labels]
    if len(atlas_labels) == args.n_rois + 1 and atlas_labels[0].lower() == "background":
        atlas_labels = atlas_labels[1:]
    network_atlas = build_network_level_atlas(atlas.maps, atlas_labels)

    sagittal_coords = evenly_spaced_coords(-50, 50, args.n_sagittal_cuts)
    coronal_coords = evenly_spaced_coords(-80, 60, args.n_coronal_cuts)
    axial_coords = evenly_spaced_coords(-30, 70, args.n_axial_cuts)

    summary_rows: list[dict[str, object]] = []

    for subject in sorted(sample["subject_id"].tolist()):
        # I check every subject separately here so the saved figures double as a record of the QC step.
        bold_path = get_forward_bold_path(derivatives_dir, subject)
        if bold_path is None:
            raise FileNotFoundError(
                f"Could not find forward preprocessed BOLD file for {subject} in {derivatives_dir}."
            )

        ortho_path = figures_dir / f"{subject}_atlas_overlay_ortho.png"
        sagittal_path = figures_dir / f"{subject}_atlas_overlay_sagittal.png"
        coronal_path = figures_dir / f"{subject}_atlas_overlay_coronal.png"
        axial_path = figures_dir / f"{subject}_atlas_overlay_axial.png"
        clean_network_path = figures_dir / f"{subject}_atlas_overlay_clean_networks.png"
        dissertation_path = figures_dir / f"{subject}_atlas_overlay_qc.png"
        t1w_path = get_mni_t1w_path(derivatives_dir, subject)

        if args.save_ortho_figure:
            save_overlay_figure(
                atlas.maps,
                subject,
                bold_path,
                ortho_path,
                "ortho",
                list(args.ortho_cut_coords),
                args.alpha,
                args.show,
            )
        if args.save_multislice_figures:
            save_overlay_figure(
                atlas.maps,
                subject,
                bold_path,
                sagittal_path,
                "x",
                sagittal_coords,
                args.alpha,
                args.show,
            )
            save_overlay_figure(
                atlas.maps,
                subject,
                bold_path,
                coronal_path,
                "y",
                coronal_coords,
                args.alpha,
                args.show,
            )
            save_overlay_figure(
                atlas.maps,
                subject,
                bold_path,
                axial_path,
                "z",
                axial_coords,
                args.alpha,
                args.show,
            )
        if args.save_clean_network_figure:
            save_clean_network_figure(
                network_atlas,
                subject,
                bold_path,
                clean_network_path,
                list(args.ortho_cut_coords),
                args.alpha,
                args.show,
            )
        if args.save_dissertation_figure:
            clean_bg_img = str(t1w_path) if t1w_path is not None else image.mean_img(str(bold_path))
            save_dissertation_network_figure(
                network_atlas,
                subject,
                clean_bg_img,
                dissertation_path,
                list(args.ortho_cut_coords),
                args.alpha,
            )

        primary_qc_path = ""
        if args.save_dissertation_figure:
            primary_qc_path = relative_project_path(dissertation_path)
        elif args.save_clean_network_figure:
            primary_qc_path = relative_project_path(clean_network_path)
        elif args.save_ortho_figure:
            primary_qc_path = relative_project_path(ortho_path)
        elif args.save_multislice_figures:
            primary_qc_path = relative_project_path(sagittal_path)

        summary_row = {
            "subject_id": subject,
            "forward_bold_file": relative_project_path(bold_path),
            "mni_t1w_file": relative_project_path(t1w_path) if t1w_path is not None else "",
            "atlas": f"Schaefer{args.n_rois}_Yeo{args.yeo_networks}",
            "atlas_resolution_mm": args.resolution_mm,
            "primary_qc_figure_file": primary_qc_path,
        }
        if args.save_ortho_figure:
            summary_row["ortho_figure_file"] = relative_project_path(ortho_path)
        if args.save_multislice_figures:
            summary_row["sagittal_figure_file"] = relative_project_path(sagittal_path)
            summary_row["coronal_figure_file"] = relative_project_path(coronal_path)
            summary_row["axial_figure_file"] = relative_project_path(axial_path)
        if args.save_clean_network_figure:
            summary_row["clean_network_figure_file"] = relative_project_path(clean_network_path)
        summary_rows.append(summary_row)

    summary_path = output_dir / "atlas_overlay_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    saved_figure_types = []
    if args.save_ortho_figure:
        saved_figure_types.append("ortho")
    if args.save_multislice_figures:
        saved_figure_types.extend(["sagittal", "coronal", "axial"])
    if args.save_clean_network_figure:
        saved_figure_types.append("clean_network")
    if args.save_dissertation_figure:
        saved_figure_types.append("main_qc")
    saved_figure_types_str = ", ".join(saved_figure_types) if saved_figure_types else "none"

    print(f"Saved {len(summary_rows)} atlas overlay QC set(s) to {figures_dir}")
    print(f"Saved figure types: {saved_figure_types_str}")
    print(f"Wrote summary TSV to {summary_path}")


if __name__ == "__main__":
    main()
