# Index


## Example 

```julia
using GGUF

model = GGUF.open_model("model.gguf")

ts = tensors(model)
ms = metadata(model)

```