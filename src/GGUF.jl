module GGUF

using Mmap, StringViews

const MmappedModel = Vector{UInt8}

include("stringviews.jl")
include("reader.jl")

include("model.jl")
include("tensor.jl")
include("metadata.jl")

end # module GGUF
