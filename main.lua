--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local valLossMSE, valLossAbs = trainer:test(0, valLoader)
   print(string.format(' * Validation Results: MSE loss is %6.3f, Average Absolute loss is %6.3f ', bestValLoss, valLossAbs))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestValLoss = math.huge
-- we also store the absLoss corresponding to the best mse validation loss
local absLoss = math.huge
local bestEpoch = 1
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local valLossMSE, valLossAbs = trainer:test(epoch, valLoader)
   print(' * MSE Validation Loss on entire Validation set is ' .. valLossMSE)
   print(' * Absolute Validation Loss on entire Validation set is ' .. valLossAbs)

   local bestModel = false
   if valLossMSE < bestValLoss then
      bestModel = true
      bestValLoss = valLossMSE
      absLoss = valLossAbs
      bestEpoch = epoch
      print(' * Best model until now is obtained at epoch # ' .. 
      epoch .. ' and has an MSE validation loss of ' .. valLossMSE .. ' and Absolute Validation loss of ' .. valLossAbs)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel)
end

print(string.format(' * Finished best MSE Validation loss is: %6.3f and occurs at epoch # %6.3f, and the corresponding Absolute Validation loss is %6.3f', bestValLoss, bestEpoch, absLoss))
