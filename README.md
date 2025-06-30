# Towards Efficient D-Core Decomposition on GPU via Load-Balanced Parallelism (VLDB submission)

## Configuration

We implement our GPU algorithms in C++ and run the experiments
on an NVIDIA GeForce RTX 4090 with 24GB of memory using
CUDA and compile with nvcc version 12.2.14. 

This project uses GNU Make 4.3 as the build system.
Make sure it is installed and available in your environment. You can check your version with:

```
make --version
```

```
GNU Make 4.3
Built for x86_64-pc-linux-gnu
```

To compile our code.

```
make
```

Then it will gereneate a execuable binary file named 'main'.

## Execute

After compile successfully, replace {dataset} with the dataset name and {algorithm id} with algorithm id.

```
./main -d {dataset} -a {algorithm id}
```

algorithm id:

- GPeel: 2
- GHI-Warp: 5
- GHI-LB: 7
- GTrim: 9

## Example:

Run GPeel on EM dataset

```
./main -d em -a 2
```

Run GHI-Warp on EM dataset

```
./main -d em -a 5
```

Run GHI-LB on EM dataset

```
./main -d em -a 7
```

Run GTrim on EM dataset

```
./main -d em -a 9
```



## Dataset

Please place the dataset inside the dataset folder. For example, if your dataset is named "em", create a subfolder within the dataset folder and store the em.txt file inside it.

Ensure that the name of the dataset folder matches the name of the .txt file.

e.g. dataset/em/em.txt





## algorithm

2. GPU-Peeling (Our proposed GPU peeling-based solution)

4. GPU-H-Index-B (Our proposed GPU h-index-based solution with binary search)

5. GPU-H-Index-L (Our proposed GPU h-index-based solution with linear search)

6. GPU-Trim (Our GPU implementation of baseline [17])

## profile

nsys profile --stats=true  ./main -d {dataset} -a {algorithm id}

ncu --section "Occupancy" ./main -d {dataset} -a {algorithm id}

