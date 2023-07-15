from config import CFG

import numpy as np
import torch


def cal_weight(values,method=0):
    """
    function:
    ---------
        为每个类别计算各自的权重.
    
    args:
    ---------
        values: 每个类别的类别数.
        method: 使用哪一种类别平衡的方法，本次实验中一共有四种，分别是:
                -------------------------------------------------------------------
                | value  |   method
                -------------------------------------------------------------------
                |   0    |   不使用任何权重，函数返回[1.0] * num_classes
                -------------------------------------------------------------------
                |   1    |   将每个类别的数目除以其中的最大值
                -------------------------------------------------------------------
                |   2    |   使用样本总数除以每个类别的数目后取对数
                -------------------------------------------------------------------
                |   3    |   使用样本总数除以每个类别的数目后不取对数直接softmax归一化
                -------------------------------------------------------------------

    returns:
    ---------
        frequency: 返回类别的权重. 
        
    """
    init_frequency = np.array(values)
    if method == 0:
        frequency = [1.0 for i in range(CFG.num_classes)]
    elif method == 1:
        frequency = np.sum(init_frequency) / init_frequency
    elif method == 2:
        frequency = np.log2(np.sum(init_frequency) / init_frequency)
    elif method == 3:
        frequency_sum = init_frequency / sum(init_frequency)
        frequency = torch.nn.functional.softmax(torch.tensor(-1*frequency_sum))
    else:
        raise ValueError("useless value")
    print("class weight: ",frequency)
    frequency = torch.tensor(frequency,dtype=torch.float)
    return frequency
    
