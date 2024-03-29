--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local lossSum = 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      print('Output value is ')
      print(output)
      print('Target value is ')
      print(self.target)
      local loss = self.criterion:forward(self.model.output, self.target)
      print('Loss value of current Batch in Training is ' .. loss) 

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      lossSum = lossSum + loss
      N = N + 1

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end
	-- return the MSE and Average Absolute Loss
   return lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()
   local lossSum = 0.0
   local absLossSum = 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      -- find MSE 
      local loss = self.criterion:forward(self.model.output, self.target)
      -- find absolute error
      -- self.target is cuda tensor and is converted to torch float tensor
      local absLoss = torch.sum(torch.abs(output - torch.FloatTensor(self.target:size()):copy(self.target)))       
      print('Validation batch output is ')
      print(output)
      print('Validation target is ')
      print(self.target)
      print('MSE Loss value of current Batch in Validation is ' .. loss)
      print('Absolute Loss value of current Batch in Validation is ' .. absLoss)

      lossSum = lossSum + loss
      absLossSum = absLossSum + absLoss
      N = N + 1

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f '):format(
         epoch, n, size, timer:time().real, dataTime))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   print((' * Finished epoch # %d \n'):format(
      epoch))
   return lossSum / N, absLossSum / N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
