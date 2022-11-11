push!(LOAD_PATH,"../src/")
using Documenter, ProximalMethods

makedocs(
    sitename="ProximalMethods.jl",
    modules=[ProximalMethods],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/qntwrsm/ProximalMethods.jl.git",
)