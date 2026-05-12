module PAMCLIReference

const DEFAULT_CT_PATH = normpath(joinpath(homedir(), "Desktop", "OBJ_0001"))
include(joinpath(dirname(@__DIR__), "src", "pam", "setup", "config.jl"))

end

function _md_cell(value)
    text = string(value)
    text = replace(text, "\n" => " ")
    text = replace(text, "|" => "\\|")
    return isempty(text) ? " " : text
end

function _md_code(value)
    text = string(value)
    isempty(text) && return " "
    return "`$(replace(text, "`" => "\\`"))`"
end

function _choices_cell(option)
    isempty(option.choices) && return " "
    return join([_md_code(choice) for choice in option.choices], ", ")
end

function _write_option_table(io, options)
    println(io, "| Option | Default | Value | Applies to | Choices | Description |")
    println(io, "|---|---:|---|---|---|---|")
    for option in options
        println(io, "| ",
            _md_code("--" * option.name), " | ",
            _md_code(option.default), " | ",
            _md_cell(option.value), " | ",
            _md_cell(option.applies_to), " | ",
            _choices_cell(option), " | ",
            _md_cell(option.description), " |")
    end
end

function generate_cli_reference(path=joinpath(@__DIR__, "src", "cli", "parameters.md"))
    options = PAMCLIReference.pam_cli_options()
    categories = unique(option.category for option in options)
    mkpath(dirname(path))

    open(path, "w") do io
        println(io, "# PAM CLI Parameters")
        println(io)
        println(io, "This page is generated from the PAM CLI option metadata in `src/pam/setup/config.jl`.")
        println(io, "Use options as `--name=value`; positional arguments are not supported.")
        println(io)
        println(io, "The listed defaults are the base defaults. `scripts/run_pam.jl` applies a few model-aware overrides after parsing:")
        println(io)
        println(io, "- `--dimension=3` defaults to `--source-model=point`, 3D coordinates, coarser `dy/dz`, and shorter `t-max-us` unless those options are provided.")
        println(io, "- `--source-model=point` defaults to coherent phase, narrower receiver aperture, shorter duration, and `--recon-bandwidth-khz=0`.")
        println(io, "- 3D `squiggle` and `network` runs default to windowed-friendly reconstruction settings such as `--recon-bandwidth-khz=40`, `--recon-window-us=40`, and `--recon-hop-us=20`.")
        println(io)
        println(io, "For practical guidance, start with [Running PAM](@ref) before tuning individual parameters.")
        println(io)

        for category in categories
            category_options = filter(option -> option.category == category, options)
            println(io, "## ", category)
            println(io)
            _write_option_table(io, category_options)
            println(io)
        end
    end

    return path
end
