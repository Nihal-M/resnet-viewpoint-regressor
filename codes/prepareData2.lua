-- reading lines from a text file and parsing it
-- below code is for preparing labels

require 'torch'   -- torch

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
