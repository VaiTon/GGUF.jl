using Documenter, GGUF

makedocs(
    modules = [GGUF],
    sitename = "GGUF.jl",
    format = Documenter.HTML()
)