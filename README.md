# DCoreGPU

## Dataset

Please place the dataset inside the dataset folder. For example, if your dataset is named "em", create a subfolder within the dataset folder and store the em.txt file inside it.

Ensure that the name of the dataset folder matches the name of the .txt file.

e.g. dataset/em/em.txt

## Compile

make

## run
./main -d dataset -a 1(e.g. ./main -d em -a 1)

## algorithm

1. klist

2. klist with prune

## profile

nsys profile --stats=true 