# @Author : Cheng Huang
# @Time   : 10:28 2024/12/3
# @File   : data_clean.py


if __name__ == '__main__':
    fw = open("/home/cheng/DCoreGPU/hollywood.txt", 'w')
    maxx = 0
    cnt = 0
    tuple_set = set()
    with open("/home/cheng/DCoreGPU/hollywood-dirty.txt") as f:
        for line in f:
            if "#" in line:
                continue
            u, v = line.rstrip().split("	")
            u = int(u)
            v = int(v)
            if u == v:
                # print(u, v)
                continue
            if (u,v) not in tuple_set:
                tuple_set.add((u, v))
                fw.write(str(u)+" "+str(v)+"\n")
    print("number of edge = ", len(tuple_set))

