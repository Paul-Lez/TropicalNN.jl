using Documenter
using TropicalNN

DocMeta.setdocmeta!(TropicalNN, :DocTestSetup, :(using TropicalNN); recursive=true)

makedocs(
    sitename = "TropicalNN.jl",
    authors = "Paul Lezeau et al.",
    modules = [TropicalNN],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/Paul-Lez/TropicalNN.jl",
    devbranch = "main",
)
