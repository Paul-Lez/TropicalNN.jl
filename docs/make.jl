using Documenter
using TropicalNN

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
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Paul-Lez/TropicalNN.jl",
    devbranch = "main",
)
