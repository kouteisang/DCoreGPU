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
./main -d {dataset} -a {algorithm id} -t {warp start}
```

algorithm id:

- GPeel: 2
- obs1: 13
- obs2: 14
- obs3: 15
- Geld-Thread: 16
- GHI-Warp: 5
- Geld-Block: 17
- Geld-Block*: 18



## Example:

Run GPeel on EM dataset

```
./main -d em -a 2
```


Run Geld on EM dataset with warp start 0

```
./main -d em -a 7 -t 0
```



## Dataset

Please place the dataset inside the dataset folder. For example, if your dataset is named "em", create a subfolder within the dataset folder and store the em.txt file inside it.

Ensure that the name of the dataset folder matches the name of the .txt file.

e.g. dataset/em/em.txt

| Dataset  |  Source |  
|---|---|
| (AM)Amazon  |  https://snap.stanford.edu/data/index.html |   
| (EM)Email-EuAll  |  https://snap.stanford.edu/data/index.html |   
| (PO)Pokec  |  https://snap.stanford.edu/data/index.html |   
| (SD)Slashdot  |  https://snap.stanford.edu/data/index.html |   
| (EN)Enwiki-2024  |  https://law.di.unimi.it/index.php |   
| (LJ)Live Journal  |  https://snap.stanford.edu/data/index.html |   
| (UK)UK-2002   |  https://law.di.unimi.it/index.php |   
| (H11)Hollywood-2011  | https://law.di.unimi.it/index.php |   
| (H09)Hollywood-2009  | https://law.di.unimi.it/index.php |   
| (IT)IT-2004  |  https://law.di.unimi.it/index.php |   
| (IC)IndoChina-2004  |  https://law.di.unimi.it/index.php |   
| (EU)EU-2015-TPD  |  https://law.di.unimi.it/index.php |   


## profile

nsys profile --stats=true  ./main -d {dataset} -a {algorithm id}

ncu --section "Occupancy" ./main -d {dataset} -a {algorithm id}

