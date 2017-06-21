
## Usage:

- RSampling: Sample input to random shape.
nn.RSampling(minShape, maxShape, diff=false)

When `diff = false`, RSampling layer will use identity transformation during test.

- AutoPadding: Adaptively Crop or pad its input to the given shape.

nn.AutoPadding(shape, diff=false)

When `diff = false`, AutoPadding layer will use identity transformation during test.

## Installation

1. Copy layers you want to `torch/extra/nn`
2. Add `require('nn.LAYERNAME')` to `torch/extra/nn/init.lua`
3. `CD` to `torch/extra/nn/` directory and run `luarocks make rocks/nn-scm-1.rockspec`
