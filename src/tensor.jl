@enum TensorType::UInt32 begin
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    # quantizations
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    # k-quantizations
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_I8
    GGML_TYPE_I16
    GGML_TYPE_I32
    GGML_TYPE_COUNT
end


struct TensorInfo
    name::StringView
    ndimensions::UInt32
    dimensions::Vector{UInt64}
    type::TensorType
    offset::UInt64
end

const HEADER_OFFSET = UInt64(25)

function read_tensor_info(m::MmappedModel, offset::UInt64)
    name, offset = read_string(m, offset)
    ndimensions = reinterpret(UInt32, m[offset:offset+3])[1]
    offset += 4

    dimensions = Vector{UInt64}(undef, ndimensions)
    for i in 1:ndimensions
        dimensions[i] = reinterpret(UInt64, m[offset:offset+7])[1]
        offset += 8
    end

    type = reinterpret(UInt32, m[offset:offset+3])[1] |> TensorType
    offset += 4

    tensor_offset = reinterpret(UInt64, m[offset:offset+7])[1]
    offset += 8

    return TensorInfo(name, ndimensions, dimensions, type, tensor_offset), offset
end

function read_tensors_info(m::MmappedModel, offset::UInt64)
    count = tensorn(m)
    infos = Vector{TensorInfo}(undef, count)

    for i in 1:count
        infos[i], offset = read_tensor_info(m, offset)
    end

    return infos, offset
end

function tensors_info(m::MmappedModel)
    _, offset = read_metadata(m, HEADER_OFFSET)  # metadata offset
    info, offset = read_tensors_info(m, offset)
    return info
end



tensors_info(m::Model) = tensors_info(m.data)

export TensorInfo, tensors_info

function tensors_offset(m::MmappedModel)
    _, offset = read_metadata(m, HEADER_OFFSET)  # metadata offset
    _, offset = read_tensors_info(m, offset)

    offset
end

function read_tensor(m::Model, info::TensorInfo)
    offset = m.tensor_offset + info.offset

    if info.type == GGML_TYPE_F32
        return reinterpret(Float32, @view m.data[offset:offset+3])
    elseif info.type == GGML_TYPE_F16
        return reinterpret(Float16, @view m.data[offset:offset+1])
    elseif info.type == GGML_TYPE_Q4_0
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q4_1
        return reinterpret(Int8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q5_0
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q5_1
        return reinterpret(Int8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q8_0
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q8_1
        return reinterpret(Int8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q2_K
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q3_K
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q4_K
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q5_K
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q6_K
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_Q8_K
        return reinterpret(UInt8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_I8
        return reinterpret(Int8, @view m.data[offset:offset+0])
    elseif info.type == GGML_TYPE_I16
        return reinterpret(Int16, @view m.data[offset:offset+1])
    elseif info.type == GGML_TYPE_I32
        return reinterpret(Int32, @view m.data[offset:offset+3])
    end

    throw(ArgumentError("Unknown tensor type: $info.type"))
end

