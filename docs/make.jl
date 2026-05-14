using Pkg

Pkg.instantiate()

using Documenter

include("generate_cli_reference.jl")
generate_cli_reference()

makedocs(
    sitename="TranscranialFUS",
    remotes=nothing,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link=nothing,
        repolink="",
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "CLI" => [
            "Overview" => "cli/overview.md",
            "Running PAM" => "cli/run-pam.md",
            "Running Focus" => "cli/run-focus.md",
            "PAM Parameters" => "cli/parameters.md",
        ],
        "Workflow" => "workflow.md",
        "Outputs" => "outputs.md",
        "Validation" => "validation.md",
        "Reference Notes" => [
            "PAM Algorithm" => "reference/pam-algorithm.md",
            "Cluster Models" => "reference/cluster-models.md",
        ],
        "Troubleshooting" => "troubleshooting.md",
    ],
    checkdocs=:none,
)
