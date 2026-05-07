function make_pam_medium_3d(
    cfg::PAMConfig3D;
    aberrator::Symbol=:none,
)
    nx = pam_Nx(cfg)
    ny = pam_Ny(cfg)
    nz = pam_Nz(cfg)
    c   = fill(Float32(cfg.c0),   nx, ny, nz)
    rho = fill(Float32(cfg.rho0), nx, ny, nz)

    if aberrator == :none || aberrator == :water
        return c, rho, Dict{Symbol, Any}(:aberrator => aberrator)
    end
    error("Aberrator type :$aberrator not supported for 3D PAM medium (only :none and :water).")
end
