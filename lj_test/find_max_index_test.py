import numpy as np


def find_max_index(y, number=3):
    '''
    找到np.array中，最大的几个数，对应的索引
    :param y:
    :param number: 索引的个数
    :return:
    '''
    index_list = []
    for i in range(number):
        index = np.argmax(y)
        if np.max(y)!=0:
            index_list.append(index)
            y[index]=0
        else:
            return index_list
    return index_list

y = np.array([3,7,0,0,0,0,0,0,0,0,9,1,3,7,10])
x = find_max_index(y,5)
pass
