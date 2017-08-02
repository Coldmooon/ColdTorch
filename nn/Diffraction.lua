------------------------------------------------------------------------
--[[ Diffraction ]]--
-- Outputs a random value given an range.
-- If nInputDim is specified, uses the input to determine the size of
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append random inputs to
-- an input : nn.ConcatTable():add(nn.Diffraction(v)):add(nn.Identity()) .
------------------------------------------------------------------------
local Diffraction, parent = torch.class("nn.Diffraction", "nn.Module")

function Diffraction:__init(min, max, nInputDim, isangle, diff, verbose)
   parent.__init(self)
   self.min = min
   self.max = max
   self.nInputDim = nInputDim
   self.isangle = isangle or false
   self.train = true
   self.diff = diff or false
   self.isverbose = verbose or false
end

function Diffraction:verbose(...)
   if self.isverbose then print('<nn.Diffraction:> ', ...) end
end

function Diffraction:updateOutput(input)

   local location = nil

   if (not self.train) and self.diff then
      location = 0
      self:verbose('Rotation is skipped during test')
   elseif self.isangle then
      location = torch.random(self.min, self.max)
      self:verbose('Picked up a random integer: ', location)
      location = location * 2 * math.pi / 360
   else
      location = torch.uniform(self.min, self.max)
      self:verbose('Picked up a random float value: ', location)
   end

   if torch.type(location) == 'number' then
      location = torch.Tensor{location}
   end
   assert(torch.isTensor(location), "Expecting number or tensor at arg 1")

   if self.nInputDim and input:dim() > self.nInputDim then
      local vsize = location:size():totable()
      self.output:resize(input:size(1), table.unpack(vsize))
      local value = location:view(1, table.unpack(vsize))
      self.output:copy(value:expand(self.output:size()))
   else
      self.output:resize(location:size()):copy(location)
   end
   return self.output
end

function Diffraction:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
end

function Diffraction:__tostring__()
   s = string.format('%s(min=%f, max=%f, nInputDim=%d, isangle=%s, diff=%s)', torch.type(self), self.min, self.max, self.nInputDim, self.isangle, self.diff)
   return s
end
