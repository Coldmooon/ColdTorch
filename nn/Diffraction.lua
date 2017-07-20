------------------------------------------------------------------------
--[[ Diffraction ]]--
-- Outputs a random value given an range.
-- If nInputDim is specified, uses the input to determine the size of
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append random inputs to
-- an input : nn.ConcatTable():add(nn.Diffraction(v)):add(nn.Identity()) .
------------------------------------------------------------------------
local Diffraction, parent = torch.class("nn.Diffraction", "nn.Module")

function Diffraction:__init(min, max, nInputDim, verbose)
   self.min = min
   self.max = max
   self.nInputDim = nInputDim
   parent.__init(self)
   self.isverbose = verbose or false
end

function Diffraction:verbose(...)
   if self.isverbose then print('<nn.Diffraction:> ', ...) end
end

function Diffraction:updateOutput(input)

   local location = torch.random(self.min, self.max)
   self:verbose('Picked up a random value: ', location)
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
   s = string.format('%s(min=%d, max=%d, nInputDim=%d)', torch.type(self), self.min, self.max, self.nInputDim)
   return s
end