--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ImagenetDataset:get(i)

	-- start of prepareData2.lua

-- reading lines from a text file and parsing it
-- below code is for preparing labels

local trsize = 8
local tesize = 8

-- create train set:
trainData = {
   labels = torch.Tensor(trsize,2),
   size = function() return trsize end
}
--create test set:
testData = {
      labels = torch.Tensor(tesize,2),
      size = function() return tesize end
   }

function read_lines(filename)
  local database = { }
  for l in io.lines(filename) do
    local i, a, b = l:match '(%S+)%s+(%S+)%s+(%S+)'
    table.insert(database, { image = i, label1=a, label2=b })
  end
  return database
end

parsedTrainFile = read_lines('/home/falak/Nihal.M/trainData.txt')
parsedTestFile = read_lines('/home/falak/Nihal.M/testData.txt')

-- we now need only a separate tensor having the labels in the same order as image1, image2 ,....

for i=1,trsize do 
   trainData.labels[i][1] = parsedTrainFile[i]["label1"]
   trainData.labels[i][2] = parsedTrainFile[i]["label2"]
end
for i=1,tesize do
   testData.labels[i][1] = parsedTestFile[i]["label1"]
   testData.labels[i][2] = parsedTestFile[i]["label2"]
end

-- end of prepareData2.lua

   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))   

   
   
-- lets parse the path to get the image no. eg if path is label1/image123.jpg, we need to separate 123 out 
-- of it to locate the corresponding label in the tensor (either testData.labels or trainData.labels)

   strParse=string.match(path,'image.*')
   index=string.match(strParse,'%d+')
   local class = torch.rand(1,2)

-- since this function :get(i) is used by both the trainer and the tester, we need to identify whether
-- labels in this case refer to train or val?? this info is present in self.dir

   print('value in self.dir is ')
   print(self.dir)
   if string.match(self.dir, 'train') == 'train' then
      class[1][1]=trainData.labels[index][1]
      class[1][2]=trainData.labels[index][2]
   else
      class[1][1]=testData.labels[index][1]
      class[1][2]=testData.labels[index][2]
   end

   print('class is ')
   print(class)
   print('path is ')
   print(path)

   return {
      input = image,
      target = class,
   }
end

function ImagenetDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ImagenetDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImagenetDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(224),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImagenetDataset
