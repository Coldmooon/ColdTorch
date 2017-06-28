local SwitcherTable, parent = torch.class('nn.SwitcherTable', 'nn.Module')

function SwitcherTable:__init(mode, inplcae, verbose)
   parent.__init(self)
   self.inplace = inplcae and false
   self.gradInput = {}
   self.index = 1
   self.res = 0
   self.mode = mode or 'mean'
   self.isverbose = verbose or false
end

function SwitcherTable:verbose(...)
   if self.isverbose then print('<nn.SwitcherTable:> ', ...) end
end

local function value_with_mode(x, mode)

   local res = nil
   if mode == 'mean' then
      res = torch.mean(x)
   elseif mode == 'max' then
   	res = torch.max(x)
   elseif mode == 'min' then
   	res = torch.min(x)
   elseif mode == 'median' then
   	res = torch.median(x)
   else
   	error('unknown mode.')
   end

   return res
end

function SwitcherTable:updateOutput(input)

   self.res = value_with_mode(input[1], self.mode)
   self.index = 1
   for i = 2, #input do
      res_tmp = value_with_mode(input[i], self.mode)
      if res_tmp > self.res then
      	self.res = res_tmp
      	self.index = i
      end
   end
   self:verbose('Forward Pass: using the stream No.' .. self.index)
   if self.inplace then
      self.output:set(input[self.index])
   else
      self.output:resizeAs(input[self.index]):copy(input[self.index])
   end

   return self.output
end

function SwitcherTable:updateGradInput(input, gradOutput)
	
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i]):fill(0.0)
      if i == self.index then
         self:verbose('Backward Pass: setting the stream No.' .. self.index .. ' to Identity')
	      if self.inplace then
	         self.gradInput[i]:set(gradOutput) -- never used
	      else
	         self.gradInput[i]:copy(gradOutput)
	      end
	  end
   end

   for i=#input+1, #self.gradInput do
      self.gradInput[i] = nil
   end

   return self.gradInput
end

function SwitcherTable:__tostring__()
   s = string.format('%s(mode=%s, inplace=%s)', torch.type(self), self.mode, self.inplace)
   return s
end