# DCoreGPU

## Dataset

Please place the dataset inside the dataset folder. For example, if your dataset is named "em", create a subfolder within the dataset folder and store the em.txt file inside it.

Ensure that the name of the dataset folder matches the name of the .txt file.

e.g. dataset/em/em.txt

## Compile

make

## run
./main -d dataset -a 1(e.g. ./main -d em -a 2)

## algorithm

2. GPU-Peeling (Our proposed GPU peeling-based solution)

4. GPU-H-Index-B (Our proposed GPU h-index-based solution with binary search)

5. GPU-H-Index-L (Our proposed GPU h-index-based solution with linear search)

6. GPU-Trim (Our GPU implementation of baseline [17])

## profile

nsys profile --stats=true  ./main -d it-2004 -a 7



nvcc -std=c++11 -o suffix suffixsum.cu 