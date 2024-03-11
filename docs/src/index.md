# Index


## Example 

```julia
using GGUF

model = GGUF.open("model.gguf")

ts = tensors(model)
ms = metadata(model)

```