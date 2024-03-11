module GGUF

using Mmap, StringViews
include("stringviews.jl")


const MmappedModel = Vector{UInt8}
"""
    Model

Represents a GGUF model. It contains the metadata and the tensor data.

See also: [`open_model`](@ref)
"""
struct Model
    data::MmappedModel
    tensor_offset::UInt64
end
export Model

"""
    open_model(file::String)

Opens a GGUF model file and returns a [`Model`](@ref) object.
"""
function open_model(file::String)
    m::MmappedModel = open(file) do f
        mmap(f, MmappedModel)
    end

    return Model(m, tensors_offset(m))
end

""" Returns the magic number of the model. """
magic(m::MmappedModel) = reinterpret(UInt32, m[1:4])[1] # 0x0000
magic(m::Model) = magic(m.data)
export magic

""" Returns the version of the model. """
version(m::MmappedModel) = reinterpret(UInt32, m[5:8])[1] # 0x0004
version(m::Model) = version(m.data)
export version

""" Returns the number of tensors in the model. """
tensorn(m::MmappedModel) = reinterpret(UInt64, m[9:16])[1] # 0x0008
tensorn(m::Model) = tensorn(m.data)
export tensorn

""" Returns the number of metadata entries in the model. """
metadatan(m::MmappedModel)::UInt64 = reinterpret(UInt64, m[17:24])[1] # 0x0010
metadatan(m::Model)::UInt64 = metadatan(m.data)
export metadatan

function Base.show(io::IO, _::MIME"text/plain", m::Model)
    if magic(m) != 0x46554747 # 0x47475546 in little endian
        @error "Not a GGUF model" magic = magic(m) expected = 0x47475546
        throw(ArgumentError("Not a GGUF model"))
    end

    println(io, "GGUF Model")

    meta = metadata(m)

    if haskey(meta, "general.name")
        println(io, "  Name: $(meta["general.name"])")
    end

    if haskey(meta, "general.architecture")
        println(io, "  Architecture: $(meta["general.architecture"])")
    end

    if haskey(meta, "general.file_type")
        println(io, "  File type: $(meta["general.file_type"])")
    end

    println(io, "  Version: $(version(m))")
    println(io, "  Tensors: $(tensorn(m))")
    println(io, "  Metadata: $(metadatan(m))")
    print(io, "  Tensor offset: $(m.tensor_offset)")

end

function align(offset::UInt64, alignment::UInt64)::UInt64
    return (offset + alignment - 1) & ~(alignment - 1)
end


function read_string(m::MmappedModel, offset::UInt64)
    len = reinterpret(UInt16, @view m[offset:offset+1])[1]
    offset += 8
    if len > 65535
        throw(ArgumentError("String length is too long"))
    end

    string_range = offset:offset+len-1
    view = StringView(@view m[string_range])

    return view, offset + len
end

include("tensor.jl")
include("metadata.jl")

end # module GGUF
