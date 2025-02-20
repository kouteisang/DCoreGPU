dataset = "slashdot"
for k in range(0, 33):
    print("k = ", k)
    ground_truth = {}
    gcore = {}

    with open("/home/cheng/PDC2/results/aa-pokec-k"+str(k)+"-cpu.txt", "r") as ft:
        for line in ft:
            v, c = line.rstrip().split(" ")
            v, c = int(v), int(c)
            ground_truth[v] = c
    ft.close()

    with open("/home/cheng/DCoreGPU/dataset/pokec/pokec-k-"+str(k)+"-gpu.txt", "r") as fg:
        for line in fg:
            v, c = line.rstrip().split(" ")
            v, c = int(v), int(c)
            gcore[v] = c
    fg.close();

    for key in gcore.keys():
        if gcore[key] != ground_truth[key]:
            print("error!")
            print("k = ", k)
            print("key = ", key)