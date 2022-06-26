push!(LOAD_PATH,"../src/")
using Documenter, FoliationsManifoldsAutoencoders
Documenter.HTML(edit_link = "main")
makedocs(
    sitename="Foliations, Manifolds and Autoencoders", 
    modules = [FoliationsManifoldsAutoencoders],
	pages = [
		"Home" => "index.md",
		"Invariant foliations" => Any[
            "Example" => "foliationexample.md",
            "Sparse foliation" => "foliationidentify.md",
            "Locally accurate foliation" => "localfoliation.md",
        ],
		"Autoencoders" => "autoencoders.md",
		"Oscillations" => "freqdamp.md",
        "Dense polynomials" => "polymethods.md",
		"Direct methods" => "invariancecalculations.md",
    ],
    format = Documenter.HTML(mathengine = Documenter.MathJax(Dict(:TeX => Dict(:equationNumbers => Dict(:autoNumber => "AMS"),
    :Macros => Dict(:ket => ["|#1\\rangle", 1],:bra => ["\\langle#1|", 1],),))), prettyurls = false, edit_link = "main"))
deploydocs(;
    repo="github.com/rs1909/FMA.git", versions = nothing,
)