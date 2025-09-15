dataset = "em"
for k in range(72, 73):
    ground_truth = {}
    gcore = {}

    with open("/home/cheng/DCoreGPU/dataset/enwiki-2024/warp-"+str(k)+".txt", "r") as ft:
        for line in ft:
            v, c = line.rstrip().split(" ")
            v, c = int(v), int(c)
            ground_truth[v] = c
    ft.close()

    with open("/home/cheng/DCoreGPU/dataset/enwiki-2024/block-"+str(k)+".txt", "r") as fg:
        for line in fg:
            v, c = line.rstrip().split(" ")
            v, c = int(v), int(c)
            gcore[v] = c
    fg.close();

    for key in ground_truth.keys():
        if gcore[key] != ground_truth[key]:
            print("error!")
            print("k = ", k)
            print("key = ", key)
            break;