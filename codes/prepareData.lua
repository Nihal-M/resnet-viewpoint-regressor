
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> loading dataset')

local trsize = 2
local tesize = 2

-- create train set:
trainData = {
   data = torch.Tensor(trsize, 3, 224, 224), -- not sure of dimensions of picture
   labels = torch.Tensor(trsize,2),
   size = function() return trsize end
}
--create test set:
testData = {
      data = torch.Tensor(tesize, 3, 224, 224),
      labels = torch.Tensor(tesize,2),
      size = function() return tesize end
   }

-- reading lines from a text file and parsing it
-- below code is for preparing labels

function read_lines(filename)
  local database = { }
  for l in io.lines(filename) do
    local i, a, b, c, d, e, f = l:match '(%S+)%s+(%S+)%s+(%S+)%s+(%S+)%s+(%S+)%s+(%S+)%s+(%S+)'
    table.insert(database, { image = i, a = a, b=b, c=c, d=d, label1=e, label2=f })
  end
  return database
end

parsedTrainFile = read_lines('/home/km/ViewpointsAndKeypoints/cachedir/VNetTrainFiles/trainData.txt')
parsedTestFile = read_lines('/home/km/ViewpointsAndKeypoints/cachedir/VNetTrainFiles/testData.txt')

-- We load the dataset and labels from disk

for i=1,trsize do
   trainData.data[i] = image.load('/home/km/ViewpointsAndKeypoints/Aseem/train/image'..i..'.png') 
   trainData.labels[i,1] = parsedTrainFile[i]["label1"]
   trainData.labels[i,2] = parsedTrainFile[i]["label2"]
end
for i=1,tesize do
   testData.data[i] = image.load('/home/km/ViewpointsAndKeypoints/Aseem/test/image'..i..'.png') 
   testData.labels[i,1] = parsedTestFile[i]["label1"]
   testData.labels[i,2] = parsedTestFile[i]["label2"]
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> preprocessing data')
-- faces and bg are already YUV here, no need to convert!

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

-- trainData.data = trainData.data:float()
-- testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
-- print '==> preprocessing data: colorspace RGB -> YUV'
-- for i = 1,trainData:size() do
--    trainData.data[i] = image.rgb2yuv(trainData.data[i])
-- end
-- for i = 1,testData:size() do
--    testData.data[i] = image.rgb2yuv(testData.data[i])
-- end

-- Name channels for convenience
local channels = {'y'}--,'u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local first256Samples_y = trainData.data[{ {1,256},1 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   local first256Samples_y = testData.data[{ {1,256},1 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
