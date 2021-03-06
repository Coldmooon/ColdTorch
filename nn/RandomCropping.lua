local RandomCropping, parent = torch.class('nn.RandomCropping', 'nn.Module')

function RandomCropping:__init(shape, diff, verbose)
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

function RandomCropping:verbose(...)
   if self.isverbose then print('<nn.RandomCropping:> ', ...) end
end

local function compute_shape(input_h, input_w, shape)
   margin_h = shape - input_h
   margin_w = shape - input_w
   if margin_w > 0 then
       pad_l = torch.random(0, margin_w) 
   else
       pad_l = torch.random(margin_w, 0)
   end
   pad_r = margin_w - pad_l

   if margin_h > 0 then
       pad_t = torch.random(0, margin_h) 
   else
       pad_t = torch.random(margin_h, 0)
   end
   pad_b = margin_h - pad_t

   -- pad_l = math.ceil(board_h)
   -- pad_t = math.ceil(board_w)
   -- pad_r = math.floor(board_h)
   -- pad_b = math.floor(board_w)

   return pad_l, pad_t, pad_r, pad_b
end

function RandomCropping:updateOutput(input)
   if (not self.train) and self.diff then 
      self:verbose('skip forward pass during test')
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
      self:verbose('auto cropping to ', h, 'x', w, 'padded by: l ', self.pad_l, ' r ', self.pad_r, ' t ', self.pad_t, ' b ', self.pad_b)
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

function RandomCropping:updateGradInput(input, gradOutput)
   if (not self.train) and self.diff then 
      self:verbose('skip backward pass during test')
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


function RandomCropping:__tostring__()
   return torch.type(self) ..
   string.format('(shape=%d, diff=%s)', self.shape, self.diff)
end
