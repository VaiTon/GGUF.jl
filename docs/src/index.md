# Index

## How to

```
julia>] add https://github.com/VaiTon/GGUF.jl
```


## Example 

```julia
using GGUF

model = GGUF.open("model.gguf")

ts = tensors(model)
ms = metadata(model)

```
