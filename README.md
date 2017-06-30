
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

This layer first computes the statistic of feature maps for each stream. 
Then, the statistics of all streams are compared through the given mode.
Finally, only the winner stream is outputted.

The statistics computed can be `max`, `min`, `mean`, or `median`.
The comparison mode can be `>(max)`, `<(min)`, or `random`

The mode is encoded by `[comparison mode + statistic]`

For example, `trainmode='maxmin'` and `testmode='randommax'` mean that compute the min value of feature maps of each stream,
and output the stream having the max statistics during training, but randomly output an stream (max is ignored) during test.

```
nn.SwitcherTable(trainmode='maxmax', testmode='maxmax', inplace=false, verbose=false)
```

```
x = torch.rand(10)
diff = torch.rand(10)

-- two streams
n = nn.ConcatTable()
n:add(nn.Linear(10,10))
n:add(nn.Linear(10,10))

m = nn.Sequential()
m:add(n):add(nn.SwitcherTable('maxmin', 'minmax'))

m:forward(x)
m:backward(x, diff)
```

# Installation

1. Copy layers you want to `torch/extra/nn`
2. Add `require('nn.LAYERNAME')` to `torch/extra/nn/init.lua`
3. `CD` to `torch/extra/nn/` directory and run `luarocks make rocks/nn-scm-1.rockspec`
