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