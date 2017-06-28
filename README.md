
# Usage:

### RSampling

Sample input to random shape.

```
nn.RSampling(minShape, maxShape, diff=false, verbose=false)
```

When `diff = false`, RSampling layer will use identity transformation during test.

### AutoPadding

Adaptively Crop or pad its input to the given shape.

```
nn.AutoPadding(shape, diff=false, verbose=false)
```

When `diff = false`, AutoPadding layer will use identity transformation during test.

### SwitcherTable

Only output the winner stream.

```
nn.SwitcherTable(mode='mean', inplace=false, verbose=false)
```

```
x = torch.rand(10)
diff = torch.rand(10)

-- two streams
n = nn.ConcatTable()
n:add(nn.Linear(10,10))
n:add(nn.Linear(10,10))

m = nn.Sequential()
m:add(n):add(nn.SwitcherTable('max'))

m:forward(x)
m:backward(x, diff)
```

# Installation

1. Copy layers you want to `torch/extra/nn`
2. Add `require('nn.LAYERNAME')` to `torch/extra/nn/init.lua`
3. `CD` to `torch/extra/nn/` directory and run `luarocks make rocks/nn-scm-1.rockspec`
