dataset = "em"
for k in range(0, 27):
    print("k = ", k)
    ground_truth = {}
    gcore = {}

    with open("/home/cheng/DCoreGPU/dataset/em/2-em-k-"+str(k)+"-gpu.txt", "r") as ft:
        for line in ft:
            v, c = line.rstrip().split(" ")
            v, c = int(v), int(c)
            ground_truth[v] = c
    ft.close()

    with open("/home/cheng/DCoreGPU/dataset/em/em-"+str(k)+"-gpu-a4.txt", "r") as fg:
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
            break;