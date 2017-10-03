
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

### RandomCropping

Randomly crop or pad its input to the given shape. This layer is similar to the AutoPadding. The difference is that the AutoPadding module perform center cropping or padding while this layer does this randomly.

```
nn.RandCropping(shape, diff=false, verbose=false)
```

### Diffraction

Sample a random value from the given distribution. This layer uses the name `Diffraction` since its function is similar to the diffraction of light. During backward, the returned gradInput is a zero Tensor of the same size as the input.

```
nn.Diffraction(min, max, nInputDim, isangle=false, diff=false, verbose=false)
```
The parameters are the following:
  * `min`: the min value of uniform distribution 
  * `max`: the max value of uniform distribution
  * `nInputDIm`: if nInputDim is specified, it uses the input to determine the size of the batch.
  * `isangle`: if isangle is `true`, the min and max will be converted to radian value.
  * `diff`: if `diff` is true, this layer will perform `Identity()` transformation during backward.
  * `verbose`: if `versbose` is true, DEBUG information will be outputted.

# Installation

1. Copy layers you want to `torch/extra/nn`
2. Add `require('nn.LAYERNAME')` to `torch/extra/nn/init.lua`
3. `CD` to `torch/extra/nn/` directory and run `luarocks make rocks/nn-scm-1.rockspec`
