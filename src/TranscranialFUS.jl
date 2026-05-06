module TranscranialFUS

using Statistics
using Random
using DICOM
using FFTW
using Interpolations
import CairoMakie
import CondaPkg
if !haskey(ENV, "SSL_CERT_FILE") && isfile("/etc/ssl/cert.pem")
    ENV["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
end
import PythonCall

const DEFAULT_CT_PATH = normpath(
    joinpath(
        @__DIR__,
        "..",
        "..",
        "Ultrasound",
        "DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)",
        "DICOM",
        "PAT_0000",
        "STD_0000",
        "SER_0002",
        "OBJ_0001",
    ),
)
const DEFAULT_ROI_INDEX_XYZ = (170, 190, 400)
const DEFAULT_ROI_SIZE_XYZ = (705, 360, 450)

export DEFAULT_CT_PATH, DEFAULT_ROI_INDEX_XYZ, DEFAULT_ROI_SIZE_XYZ
export CTInfo, KGrid2D, SimulationConfig, SweepSettings, AnimationSettings, MediumType, Est
export WATER, SKULL_IN_WATER, GEOMETRIC, HASA
export parse_placement_mode, resolve_placement_mode
export omega, Nx, Nz, Nt, Nx_hasa, target_index, Nz_active, active_col_range, set_z_focus!
export load_roi_resample_xy, load_default_ct
export hu_to_rho_c, find_skull_boundaries, skull_mask_from_c_columnwise
export make_medium_fixed_distance_from_skull, make_medium_fixed_transducer, make_medium
export plot_hasa_results, focus, analyse_focus_2d, run_focus_case, kwave_available
export EmissionSource2D, PointSource2D, BubbleCluster2D, GaussianPulseCluster2D, StochasticSource2D, PAMConfig, PAMWindowConfig, SourceVariabilityConfig, fit_pam_config, pam_Nx, pam_Ny, pam_Nt, pam_grid, receiver_row, receiver_col_range, depth_coordinates
export emission_frequencies, cavitation_model
export make_vascular_bubble_clusters, make_burst_train_sources, make_pam_medium, source_grid_index, simulate_point_sources, reconstruct_pam, reconstruct_pam_windowed, find_pam_peaks, find_pam_peaks_clean
export pam_truth_mask, pam_centerline_truth_mask, pam_source_map, pam_psf_blur, pam_psf_blurred_truth_map, threshold_pam_map, pam_intensity_metrics, analyse_pam_2d, analyse_pam_detection_2d, reconstruct_pam_case, run_pam_case, run_pam_sweep

include("ct.jl")
include("focus.jl")
include("medium.jl")
include("pam.jl")
include("kwave_wrapper.jl")
include("analysis.jl")

end
