module TranscranialFUS

using Statistics
using Random
using DICOM
using FFTW
using Interpolations
import CairoMakie
import CondaPkg
import CUDA
if !haskey(ENV, "SSL_CERT_FILE") && isfile("/etc/ssl/cert.pem")
    ENV["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
end
import PythonCall

const DEFAULT_CT_PATH = normpath(joinpath(homedir(), "Desktop", "OBJ_0001"))
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
export EmissionSource2D, PointSource2D, BubbleCluster2D, PAMConfig, PAMWindowConfig, SourceVariabilityConfig, fit_pam_config, pam_Nx, pam_Ny, pam_Nt, pam_grid, receiver_row, receiver_col_range, depth_coordinates
export emission_frequencies
export make_squiggle_bubble_sources, make_pam_medium, source_grid_index, simulate_point_sources, simulate_point_sources_3d, reconstruct_pam, reconstruct_pam_windowed, find_pam_peaks
export pam_truth_mask, pam_centerline_truth_mask, pam_source_map, pam_psf_blur, pam_psf_blurred_truth_map, threshold_pam_map, pam_intensity_metrics, analyse_pam_2d, analyse_pam_detection_2d, reconstruct_pam_case, run_pam_case, run_pam_sweep
export EmissionSource3D, PointSource3D, BubbleCluster3D, make_squiggle_bubble_sources_3d, make_network_bubble_sources_3d
export PAMConfig3D, pam_Nz, pam_grid_3d, receiver_col_range_y, receiver_col_range_z, depth_coordinates_3d, fit_pam_config_3d, source_grid_index_3d
export make_pam_medium_3d
export PAMCUDASetup3D, reconstruct_pam_3d, reconstruct_pam_windowed_3d
export find_pam_peaks_3d, pam_truth_mask_3d, source_detection_stats_3d, threshold_detection_stats_3d, best_threshold_entry_3d, threshold_outline_entries_3d, analyse_pam_3d

include("focus.jl")
include("pam.jl")

end
