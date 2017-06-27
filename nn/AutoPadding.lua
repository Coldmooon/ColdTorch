local AutoPadding, parent = torch.class('nn.AutoPadding', 'nn.Module')

function AutoPadding:__init(shape, diff, verbose)
   parent.__init(self)
   self.shape = shape
   self.pad_l = 0
   self.pad_r = 0
   self.pad_t = 0
   self.pad_b = 0
   self.train = true
   self.diff  = diff or false
   self.isverbose = verbose or false
end

function AutoPadding:verbose(...)
   if self.isverbose then print('<nn.AutoPadding:> ', ...) end
end

local function compute_shape(input_h, input_w, shape)
   board_h = (shape - input_h)/2
   board_w = (shape - input_w)/2

   pad_l = math.ceil(board_h)
   pad_t = math.ceil(board_w)
   pad_r = math.floor(board_h)
   pad_b = math.floor(board_w)

   return pad_l, pad_t, pad_r, pad_b
end

function AutoPadding:updateOutput(input)
   if (not self.train) and self.diff then 
      self:verbose('skip forward in AutoPadding')
      self.output:resizeAs(input):copy(input)
      return self.output
   end
   if input:dim() == 3 then

      local input_h = input:size(2)
      local input_w = input:size(3)
      self.pad_l, self.pad_t, self.pad_r, self.pad_b = compute_shape(input_h, input_w, self.shape)

      -- sizes
      local h = input:size(2) + self.pad_t + self.pad_b
      local w = input:size(3) + self.pad_l + self.pad_r
      if h ~= self.shape or w ~= self.shape then 
         print('input shape:')
         print('input height: ', input_h, 'input width: ', input_w)
         print('computed board: ')
         print('board_h: ', board_h, 'board_w: ', board_w)
         print('computed shape: ')
         print('self.pad_l: ', self.pad_l)
         print('self.pad_t: ', self.pad_t)
         print('self.pad_r: ', self.pad_r)
         print('self.pad_b: ', self.pad_b)
         error('output shape computed wrong') 
      end
      if w < 1 or h < 1 then error('input is too small') end
      self.output:resize(input:size(1), h, w)
      self.output:zero()
      -- crop input if necessary
      local c_input = input
      if self.pad_t < 0 then c_input = c_input:narrow(2, 1 - self.pad_t, c_input:size(2) + self.pad_t) end
      if self.pad_b < 0 then c_input = c_input:narrow(2, 1, c_input:size(2) + self.pad_b) end
      if self.pad_l < 0 then c_input = c_input:narrow(3, 1 - self.pad_l, c_input:size(3) + self.pad_l) end
      if self.pad_r < 0 then c_input = c_input:narrow(3, 1, c_input:size(3) + self.pad_r) end
      -- crop outout if necessary
      local c_output = self.output
      if self.pad_t > 0 then c_output = c_output:narrow(2, 1 + self.pad_t, c_output:size(2) - self.pad_t) end
      if self.pad_b > 0 then c_output = c_output:narrow(2, 1, c_output:size(2) - self.pad_b) end
      if self.pad_l > 0 then c_output = c_output:narrow(3, 1 + self.pad_l, c_output:size(3) - self.pad_l) end
      if self.pad_r > 0 then c_output = c_output:narrow(3, 1, c_output:size(3) - self.pad_r) end
      -- copy input to output
      c_output:copy(c_input)
   elseif input:dim() == 4 then
      local input_h = input:size(3)
      local input_w = input:size(4)
      self.pad_l, self.pad_t, self.pad_r, self.pad_b = compute_shape(input_h, input_w, self.shape)

      -- sizes
      local h = input:size(3) + self.pad_t + self.pad_b
      local w = input:size(4) + self.pad_l + self.pad_r
      if h ~= self.shape or w ~= self.shape then
         print('input shape:')
         print('input height: ', input_h, 'input width: ', input_w)
         print('computed board: ')
         print('board_h: ', board_h, 'board_w: ', board_w)
         print('computed shape: ')
         print('self.pad_l: ', self.pad_l)
         print('self.pad_t: ', self.pad_t)
         print('self.pad_r: ', self.pad_r)
         print('self.pad_b: ', self.pad_b)
         error('output shape computed wrong') 
      end
      if w < 1 or h < 1 then error('input is too small') end
      self:verbose('auto cropping to ', h, 'x', w)
      self.output:resize(input:size(1), input:size(2), h, w)
      self.output:zero()
      -- crop input if necessary
      local c_input = input
      if self.pad_t < 0 then c_input = c_input:narrow(3, 1 - self.pad_t, c_input:size(3) + self.pad_t) end
      if self.pad_b < 0 then c_input = c_input:narrow(3, 1, c_input:size(3) + self.pad_b) end
      if self.pad_l < 0 then c_input = c_input:narrow(4, 1 - self.pad_l, c_input:size(4) + self.pad_l) end
      if self.pad_r < 0 then c_input = c_input:narrow(4, 1, c_input:size(4) + self.pad_r) end
      -- crop outout if necessary
      local c_output = self.output
      if self.pad_t > 0 then c_output = c_output:narrow(3, 1 + self.pad_t, c_output:size(3) - self.pad_t) end
      if self.pad_b > 0 then c_output = c_output:narrow(3, 1, c_output:size(3) - self.pad_b) end
      if self.pad_l > 0 then c_output = c_output:narrow(4, 1 + self.pad_l, c_output:size(4) - self.pad_l) end
      if self.pad_r > 0 then c_output = c_output:narrow(4, 1, c_output:size(4) - self.pad_r) end
      -- copy input to output
      c_output:copy(c_input)
   else
      error('input must be 3 or 4-dimensional')
   end
   return self.output
end

function AutoPadding:updateGradInput(input, gradOutput)
   if (not self.train) and self.diff then 
      self:verbose('skip backward in AutoPadding')
      -- self.gradInput = gradOutput
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      return self.gradInput
   end
   if input:dim() == 3 then
      self.gradInput:resizeAs(input):zero()
      -- crop gradInput if necessary
      local cg_input = self.gradInput
      if self.pad_t < 0 then cg_input = cg_input:narrow(2, 1 - self.pad_t, cg_input:size(2) + self.pad_t) end
      if self.pad_b < 0 then cg_input = cg_input:narrow(2, 1, cg_input:size(2) + self.pad_b) end
      if self.pad_l < 0 then cg_input = cg_input:narrow(3, 1 - self.pad_l, cg_input:size(3) + self.pad_l) end
      if self.pad_r < 0 then cg_input = cg_input:narrow(3, 1, cg_input:size(3) + self.pad_r) end
      -- crop gradOutout if necessary
      local cg_output = gradOutput
      if self.pad_t > 0 then cg_output = cg_output:narrow(2, 1 + self.pad_t, cg_output:size(2) - self.pad_t) end
      if self.pad_b > 0 then cg_output = cg_output:narrow(2, 1, cg_output:size(2) - self.pad_b) end
      if self.pad_l > 0 then cg_output = cg_output:narrow(3, 1 + self.pad_l, cg_output:size(3) - self.pad_l) end
      if self.pad_r > 0 then cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_r) end
      -- copy gradOuput to gradInput
      cg_input:copy(cg_output)
   elseif input:dim() == 4 then
      self.gradInput:resizeAs(input):zero()
      -- crop gradInput if necessary
      local cg_input = self.gradInput
      if self.pad_t < 0 then cg_input = cg_input:narrow(3, 1 - self.pad_t, cg_input:size(3) + self.pad_t) end
      if self.pad_b < 0 then cg_input = cg_input:narrow(3, 1, cg_input:size(3) + self.pad_b) end
      if self.pad_l < 0 then cg_input = cg_input:narrow(4, 1 - self.pad_l, cg_input:size(4) + self.pad_l) end
      if self.pad_r < 0 then cg_input = cg_input:narrow(4, 1, cg_input:size(4) + self.pad_r) end
      -- crop gradOutout if necessary
      local cg_output = gradOutput
      if self.pad_t > 0 then cg_output = cg_output:narrow(3, 1 + self.pad_t, cg_output:size(3) - self.pad_t) end
      if self.pad_b > 0 then cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_b) end
      if self.pad_l > 0 then cg_output = cg_output:narrow(4, 1 + self.pad_l, cg_output:size(4) - self.pad_l) end
      if self.pad_r > 0 then cg_output = cg_output:narrow(4, 1, cg_output:size(4) - self.pad_r) end
      -- copy gradOuput to gradInput
      cg_input:copy(cg_output)
   else
      error('input must be 3 or 4-dimensional')
   end
   return self.gradInput
end


function AutoPadding:__tostring__()
   return torch.type(self) ..
   string.format('(shape=%d, diff=%s)', self.shape, self.diff)
end
