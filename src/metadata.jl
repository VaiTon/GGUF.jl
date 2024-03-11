
const MetadataPrimitiveValue = Union{UInt8,Int8,UInt16,Int16,UInt32,Int32,Float32,Bool,StringView,UInt64,Int64,Float64}
const MetadataKey = StringView
const MetadataValue = Union{MetadataPrimitiveValue,Vector{T} where T<:MetadataPrimitiveValue}
const MetadataDict = Dict{MetadataKey,MetadataValue}

function read_metadata(m::MmappedModel, offset::UInt64)
    count = metadatan(m)
    meta = MetadataDict()
    sizehint!(meta, count)

    for _ in 0:(count-1)
        key, offset = read_string(m, offset)
        type, offset = read_metadata_value_type(m, offset)
        value, offset = read_metadata_value(m, offset, type)
        meta[key] = value
    end

    return meta, offset
end


""" 
Represents the type of a metadata value. 

See [`metadata`](@ref) for more information.
"""
@enum MetadataValueType::UInt32 begin
    # The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0
    # The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1
    # The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2
    # The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3
    # The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4
    # The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5
    # The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
    # The value is a boolean.
    # 1-byte value where 0 is false and 1 is true.
    # Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7
    # The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8
    # The value is an array of other values, with the length and type prepended.
    #/
    # Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9
    # The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10
    # The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11
    # The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12
end


function read_metadata_value(m::MmappedModel, offset::UInt64, type::MetadataValueType)
    if type == GGUF_METADATA_VALUE_TYPE_UINT8
        value = reinterpret(UInt8, m[offset:offset+0])[1]
        offset += 1
    elseif type == GGUF_METADATA_VALUE_TYPE_INT8
        value = reinterpret(Int8, m[offset:offset+0])[1]
        offset += 1
    elseif type == GGUF_METADATA_VALUE_TYPE_UINT16
        value = reinterpret(UInt16, m[offset:offset+1])[1]
        offset += 2
    elseif type == GGUF_METADATA_VALUE_TYPE_INT16
        value = reinterpret(Int16, m[offset:offset+1])[1]
        offset += 2
    elseif type == GGUF_METADATA_VALUE_TYPE_UINT32
        value = reinterpret(UInt32, m[offset:offset+3])[1]
        offset += 4
    elseif type == GGUF_METADATA_VALUE_TYPE_INT32
        value = reinterpret(Int32, m[offset:offset+3])[1]
        offset += 4
    elseif type == GGUF_METADATA_VALUE_TYPE_FLOAT32
        value = reinterpret(Float32, m[offset:offset+3])[1]
        offset += 4
    elseif type == GGUF_METADATA_VALUE_TYPE_BOOL
        value = reinterpret(UInt8, m[offset:offset+0])[1] == 1
        offset += 1
    elseif type == GGUF_METADATA_VALUE_TYPE_STRING
        value, offset = read_string(m, offset)
    elseif type == GGUF_METADATA_VALUE_TYPE_ARRAY
        type, offset = read_metadata_value_type(m, offset)

        len = reinterpret(UInt64, m[offset:offset+7])[1]
        offset += 8

        if len == 0
            throw(ArgumentError("Array length is 0"))
        end

        value, offset = read_metadata_value(m, offset, type)
        value = Vector{typeof(value)}(undef, len)

        for i in 2:len
            v, offset = read_metadata_value(m, offset, type)
            value[i] = v
        end
    elseif type == GGUF_METADATA_VALUE_TYPE_UINT64
        value = reinterpret(UInt64, m[offset:offset+7])[1]
    elseif type == GGUF_METADATA_VALUE_TYPE_INT64
        value = reinterpret(Int64, m[offset:offset+7])[1]
    elseif type == GGUF_METADATA_VALUE_TYPE_FLOAT64
        value = reinterpret(Float64, m[offset:offset+7])[1]
    else
        throw(ArgumentError("Unknown metadata type: $type"))
    end

    return value, offset
end


function read_metadata_value_type(m::MmappedModel, offset::UInt64)
    type = reinterpret(UInt32, m[offset:offset+3])[1] |> MetadataValueType
    offset += 4

    return type, offset
end

"""
    metadata(m::Model)

Returns the metadata of the given model. 

The metadata is a dictionary with the keys being the metadata keys and the values being the metadata values.

The metadata values can be of the following types:
- UInt8
- UInt16
- UInt32
- UInt64
- Int8
- Int16
- Int32
- Int64
- Float32
- Float64
- Bool
- String
- Array of the above types

Types are defined in the [`MetadataValueType`](@ref) enum.

"""
function metadata(m::MmappedModel)
    offset = UInt64(25)
    meta, offset = read_metadata(m, offset)
    return meta
end

metadata(m::Model) = metadata(m.data)

export metadata


