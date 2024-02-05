module GGUF

using Mmap, StringViews
include("stringviews.jl")


const MmappedModel = Vector{UInt8}

struct Model
    data::MmappedModel
    tensor_offset::UInt64
end

function open_model(file::String)
    f = open(file)
    m::MmappedModel = mmap(f, MmappedModel)
    close(f)

    return Model(m, tensors_offset(m))
end

magic(m::MmappedModel) = reinterpret(UInt32, m[1:4])[1] # 0x0000
magic(m::Model) = magic(m.data)

version(m::MmappedModel) = reinterpret(UInt32, m[5:8])[1] # 0x0004
version(m::Model) = version(m.data)

tensorn(m::MmappedModel) = reinterpret(UInt64, m[9:16])[1] # 0x0008
tensorn(m::Model) = tensorn(m.data)

metadatan(m::MmappedModel)::UInt64 = reinterpret(UInt64, m[17:24])[1] # 0x0010
metadatan(m::Model)::UInt64 = metadatan(m.data)

export magic, version, tensorn, metadatan

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
include("tensor.jl")

end # module GGUF
