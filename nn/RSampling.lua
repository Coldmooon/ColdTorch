require 'nn.THNN'
local RSampling, parent = torch.class('nn.RSampling', 'nn.Module')

function RSampling:__init(minShape, maxShape, diff)
   parent.__init(self)
   self.owidth, self.oheight = nil, nil
   self.minShape = minShape
   self.maxShape = maxShape
   self.inputSize = torch.LongStorage(4)
   self.outputSize = torch.LongStorage(4)
   self.train = true
   self.diff = diff or false
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
         self._gradOutput = self._gradOutput or gradOutput.new()
         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
         gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function RSampling:setSize(input)
   local xdim = input:dim()
   local ydim = xdim - 1
   for i = 1, input:dim() do
      self.inputSize[i] = input:size(i)
      self.outputSize[i] = input:size(i)
   end

   self.outputSize[ydim] = self.oheight
   self.outputSize[xdim] = self.owidth

end

function RSampling:updateOutput(input)

   if (not self.train) and self.diff then
      print('skip forward in RSampling')
      self.output:resizeAs(input):copy(input)
   else 
      s = torch.random(self.minShape, self.maxShape)
      self.owidth, self.oheight = s, s
      print('random shape: ', s)
      assert(input:dim() == 4 or input:dim()==3,
               'RSampling only supports 3D or 4D tensors' )
      input = makeContiguous(self, input)
      local inputwas3D = false
      if input:dim() == 3 then
         input=input:view(-1, input:size(1), input:size(2), input:size(3))
         inputwas3D = true
      end
      local xdim = input:dim()
      local ydim = xdim - 1
      self:setSize(input)
      input.THNN.SpatialUpSamplingBilinear_updateOutput(
         input:cdata(),
         self.output:cdata(),
         self.outputSize[ydim],
         self.outputSize[xdim]
      )
      if inputwas3D then
         input = input:squeeze(1)
         self.output = self.output:squeeze(1)
      end
   end
   return self.output
end

function RSampling:updateGradInput(input, gradOutput)

   if (not self.train) and self.diff then
      print('skip backward in RSampling')
      self.gradInput = gradOutput  
   else 
      assert(input:dim() == 4 or input:dim()==3,
               'RSampling only support 3D or 4D tensors' )
      assert(input:dim() == gradOutput:dim(),
        'Input and gradOutput should be of same dimension' )
      input, gradOutput = makeContiguous(self, input, gradOutput)
      local inputwas3D = false
      if input:dim() == 3 then
         input = input:view(-1, input:size(1), input:size(2), input:size(3))
         gradOutput = gradOutput:view(-1, gradOutput:size(1), gradOutput:size(2),
                  gradOutput:size(3))
         inputwas3D = true
      end
      local xdim = input:dim()
      local ydim = xdim - 1
      self.gradInput:resizeAs(input)   
      input.THNN.SpatialUpSamplingBilinear_updateGradInput(
         gradOutput:cdata(),
         self.gradInput:cdata(),
         input:size(1),
         input:size(2),
         input:size(3),
         input:size(4),
         self.outputSize[ydim],
         self.outputSize[xdim]
      )
      if inputwas3D then
         input = input:squeeze(1)
         gradOutput = gradOutput:squeeze(1)
         self.gradInput = self.gradInput:squeeze(1)
      end
   end

   return self.gradInput
end
