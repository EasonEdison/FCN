import os
import numpy as np
import random
import scipy.misc

root = "CamVid/"
all_data_dir    = os.path.join(root, "701_StillsRaw_full")
label_dir       = os.path.join(root, "LabeledApproved_full")
label_colors    = os.path.join(root, "label_colors.txt")
label_index_dir = os.path.join(root, "label_index")
train_list_csv  = os.path.join(root, "train_list.csv")
val_list_csv    = os.path.join(root, "val_list.csv")



def devide_train_val(val_rate=0.1, shuffle=True, random_seed=None):#random_seed改成None好一点，可能不需要固定
    #可以把一些路径，名字，都放一块
    all_list    = os.listdir(all_data_dir)
    val_len     = int(val_rate*len(all_list)) #wrong 这里忘了int，70.1是没办法切片的
    list_len    = len(all_list)
    #print("list_len",val_len)
    #此处应该先判断一下，有没有设置random_seed wrong
    if random_seed:
        random.seed = random_seed
    if shuffle:
        #wrong
        #报错1：TypeError: sample() missing 1 required positional argument: 'k'
        #括号写错了，导致参数不对
        #而且这里不应是把all_list打乱顺序，只需要有打乱顺序的index就可以了
        list_index    = random.sample(range(list_len), list_len) #这里处理后返回的是一个list
    else:
        list_index = list(range(list_len))  # wrong 注意这里，list才能执行切片的操作

    #此处的分配有问题:应该是获取list的元素，获取的方式是list[index]，这个放到for中
    #print("list_index",list_index)
    train_list  = [all_list[i] for i in list_index[val_len:]]
    val_list  = [all_list[i] for i in list_index[:val_len]]


    w_t = open(train_list_csv,'w')
    w_t.write("image,label:\n")
    for i in train_list:
        image_name = i
        label_name = i.split('.')[0] +'_l.png'
        w_t.write("{},{}\n".format(image_name,label_name)) #wrong 这里忘了换行符了
    w_t.close()

    w_v = open(val_list_csv,'w')
    w_v.write("image,label:\n")
    for i in val_list:
        image_name = i
        label_name = i.split('.')[0] +'_l.png'
        w_v.write("{},{}\n".format(image_name,label_name))
    w_v.close()


#字典是有key的

color2index = {}


def parse_label():
    #wrong 确实要忽略最后一个\n后面的那个元素
    color_txt = open(label_colors,'r').read().split('\n')[:-1] #wrong 此处忘了.read(),有这个才能获取字符串
    #wrong,这里没用可迭代的enumerate,用这个才能返回两个参数
    #print("color_txt",color_txt)
    for index,line in enumerate(color_txt):
        label = line.split()[-1]
        #wrong 没加int,导致color里面都是string
        color = tuple( int(x) for x in line.split()[:-1])

        color2index[color] = index


    #准备工作结束，开始解析每个label，将颜色信息换成index信息
    #先检查原来是否存在，已经解析过的就不处理了
    #print("color idnex",color2index)
    label_list = os.listdir(label_dir)
    #此处返回index
    for index,label_name in enumerate(label_list):
        if os.path.exists(label_index_dir)!=True:
            os.makedirs(label_index_dir)
        #wrong 地址搞错了
        label_save_path = os.path.join(label_dir,label_name)
        label_npy_save_path = os.path.join(label_index_dir,label_name)

        if os.path.exists(label_npy_save_path+".npy"):
            print("Haven done,skip {}".format(label_name))
            continue
        #wrong 没加mode='RGB',没加的话就是默认的'P'模式，返回2-D的输出，而不是3-D的
        label_mat = scipy.misc.imread(label_save_path,mode='RGB')
        h,w,_=label_mat.shape
        #print("shape is ",label_mat.shape)
        #初始化为0，这样即使有识别不出来的也没关系
        label_index_mat = np.zeros((h,w))
        #print("corlor2index",color2index)
        print("Start parse {}".format(label_name))
        for i in range(h):
            for j in range(w):
                try:
                    color = tuple(label_mat[i,j])
                    #print("color",color)
                    #print("index",color2index[color])
                    ind_mat = color2index[color]
                    #print("labe_mat",label_mat[i,j,:])
                    #print("ind_mat",ind_mat)
                    label_index_mat[i,j] = ind_mat
                    #("do")
                except:
                    #True
                    print("error")
        #wrong 没把label_index_mat改成Int8格式
        label_index_mat = label_index_mat.astype(np.uint8)
        #wrong 这里只给了名字，没给矩阵
        np.save(label_npy_save_path,label_index_mat)
        print("Finish {}".format(label_name))

    #check 现在还不知道啥对应啥，就不写了





if __name__ == '__main__':
    devide_train_val(val_rate=0.1, shuffle=True, random_seed=2)
    parse_label()