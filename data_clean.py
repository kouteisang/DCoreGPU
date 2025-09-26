# # @Author : Cheng Huang
# # @Time   : 10:28 2024/12/3
# # @File   : data_clean.py
# from tqdm import tqdm


# if __name__ == '__main__':
#     fw = open("/home/cheng/DCoreGPU/dataset/CollegeMsg-June/CollegeMsg-June-new.txt", 'w')
#     maxx = 0
#     cnt = 0
#     tuple_set = set()
#     with open("/home/cheng/DCoreGPU/dataset/CollegeMsg-June/CollegeMsg-June.txt") as f:
#         for line in tqdm(f):
#             if "#" in line:
#                 continue
#             u, v = line.rstrip().split(" ")
#             u = int(u)
#             v = int(v)
#             if u == v:
#                 # print(u, v)
#                 continue
#             if (u,v) not in tuple_set:
#                 maxx = max(maxx, u)
#                 maxx = max(maxx, v)
#                 tuple_set.add((u, v))
#                 fw.write(str(u)+" "+str(v)+"\n")
#     print("number of vertex = ", maxx + 1)
#     print("number of edge = ", len(tuple_set))

