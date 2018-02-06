local SwitcherTable, parent = torch.class('nn.SwitcherTable', 'nn.Module')

function SwitcherTable:__init(trainmode, testmode, inplcae, verbose)
   parent.__init(self)
   self.inplace = inplcae and false
   self.gradInput = {}
   self.index = 1
   self.trainmode = trainmode or 'maxmax'
   self.testmode = testmode or 'maxmax'
   self.train = true
   self.isverbose = verbose or false
end

function SwitcherTable:verbose(...)
   if self.isverbose then print('<nn.SwitcherTable:> ', ...) end
end

local function statistic(x, mode)

   local res = nil
   if mode == 'mean' then
      -- print('computing the statistics of feature maps [mean]')
      res = torch.mean(x)
   elseif mode == 'max' then
      -- print('computing the statistics of feature maps [max]')
      res = torch.max(x)
   elseif mode == 'min' then
      -- print('computing the statistics of feature maps [min]')
      -- local y = torch.sort(torch.abs(x):view(1,-1))
      -- local id = y:nonzero() -- remove zeros
      -- res = y[id[1][1]][id[1][2]]
      res = torch.min(x)
   elseif mode == 'median' then
      -- print('computing the statistics of feature maps [median]')
      local y = x:view(1,-1):float() -- copy to CPU
      res = torch.median(y) -- get median
      res = torch.min(res) -- convert tensor to number type
   elseif mode == 'norm' then
      -- print('computing the statistics of feature maps [norm]')
      res = torch.norm(x)
   else
      error('unknown mode.')
   end

   return res
end

-- not used yet
local function dual_operator(op)

   local dual_op = nil

   if op == 'max' then
      dual_op = 'min'
   elseif op == 'min' then
      dual_op = 'max'
   elseif op == 'mean' then
      dual_op = 'mean'
   else
      error('unknown operator')
   end
   
   return dual_op
end

local function assign_op(mode)

   local to_compare = 'max'
   local op = 'max'

   if mode == 'maxmax' then
      to_compare = 'max'
      op = 'max'
   elseif mode == 'maxmin' then
      to_compare = 'max'
      op = 'min'
   elseif mode == 'maxmean' then
      to_compare = 'max'
      op = 'mean'
   elseif mode == 'maxmedian' then
      to_compare = 'max'
      op = 'median'
   elseif mode == 'maxnorm' then
      to_compare = 'max'
      op = 'norm'

   elseif mode == 'minmax' then
      to_compare = 'min'
      op = 'max'
   elseif mode == 'minmin' then
      to_compare = 'min'
      op = 'min'
   elseif mode == 'minmean' then
      to_compare = 'min'
      op = 'mean'
   elseif mode == 'minmedian' then
      to_compare = 'min'
      op = 'median'
   elseif mode == 'minnorm' then
      to_compare = 'min'
      op = 'norm'

   elseif mode == 'medianmax' then
      to_compare = 'median'
      op = 'max'
   elseif mode == 'medianmin' then
      to_compare = 'median'
      op = 'min'
   elseif mode == 'medianmean' then
      to_compare = 'median'
      op = 'mean'
   elseif mode == 'medianmedian' then
      to_compare = 'median'
      op = 'median'
   elseif mode == 'mediannorm' then
      to_compare = 'median'
      op = 'norm'

   elseif mode:find('random') then
      to_compare = 'random'
      op = ''
   else
      error('unknown mode')
   end

   return to_compare, op

end

local function contest(input, compare_mode, op_for_feature)

   if compare_mode == 'random' then
      -- print('randomly output an stream without computing the statistics')
      return torch.random(1, #input)
   end
   
   if compare_mode == 'median' then   
      local stats = torch.zeros(#input)
      for i = 1, #input do
         stats[i] = statistic(input[i], op_for_feature)      
      end
      
      sorted_stats, positions = stats:sort()
      winner = positions[torch.ceil(#input/2)]

      return winner
   end

   local res = statistic(input[1], op_for_feature)
   local winner = 1

   for i = 2, #input do
      local res_tmp = statistic(input[i], op_for_feature)
      if compare_mode == 'max' then
         -- print('compare the statistics of feature maps among streams by max')
         if res_tmp > res then
            res = res_tmp
            winner = i
         end
      elseif compare_mode == 'min' then 
         --  print('compare the statistics of feature maps among streams by min')
         if res_tmp < res then
            res = res_tmp
            winner = i
         end
      else
          error('unknown comparison mode')
      end
   end

   return winner

end

function SwitcherTable:updateOutput(input)
   
   local compare_mode = nil
   local op_for_feautre = nil

   if self.train then
      compare_mode, op_for_feature = assign_op(self.trainmode)
      self:verbose(string.format('using %s%s at training stage', compare_mode, op_for_feature))
   else
      compare_mode, op_for_feature = assign_op(self.testmode)
      self:verbose(string.format('using %s%s at test stage', compare_mode, op_for_feature))
   end

   self.index = contest(input, compare_mode, op_for_feature)
   self:verbose('Forward Pass: output the stream No.' .. self.index)

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
   s = string.format('%s(trainmode=%s, testmode=%s, inplace=%s)', torch.type(self), self.trainmode, self.testmode, self.inplace)
   return s
end
