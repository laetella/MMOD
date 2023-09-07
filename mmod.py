# -*- coding: utf-8 -*-

import os, arff
from math import sqrt
import matplotlib.pyplot as plt  
from numpy import mean, std, array, count_nonzero
from scipy.io import loadmat
from sklearn.metrics import auc, precision_score, recall_score, f1_score
import numpy as np

def dist(point1,point2):
    sum_dis = 0.0
    dimension = len(point1)
    for index in range(dimension)  :
        sum_dis += (point2[index] - point1[index])**2
    return sqrt(sum_dis)

def loadData(fileName,data_type, str): 
    point_set = [] 
    for line in open(fileName, 'r'): 
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return array(point_set) 

def plt_outliers(point_set, clusters, fileName):
    plt.figure(0)
    for i, point in enumerate(point_set) :
        # plt.annotate(point_set.index(point), xy = (point[0], point[1]),xycoords = 'data',fontsize=3)
        if clusters[i] == 1 :
            plt.scatter(point[0],point[1],color= 'r', marker = '*')
        else:
            plt.scatter(point[0],point[1],color= 'b', marker = 'o')
    plt.savefig('../result/%s_our.png'%(fileName.split('/')[2].split('.')[0]), dpi=200)
    plt.close(0)

def get_mean_std(edge_set):
    sum_dist = 0; std = 0
    n = len(edge_set )
    for edge in edge_set:
        sum_dist += edge[2]
    mean = sum_dist / n
    for edge in edge_set:
        std += abs(edge[2] - mean) ** 2
    std = sqrt(std)
    return mean

def update_nodes(point_set, s, noise_th, nodes_finished, nodes_unfinished):
    result_set = []
    nodes_finished.append(s)
    nodes_unfinished.remove(s)
    ratio_arr = [1] * len(point_set)
    dist_arr = [0] * len(point_set)
    edge_arr = [-1]* len(point_set)
    temp_dist1 = 1.0e14; position = -1
    for index in range(len(point_set)):
        if index == s :
            continue
        t = dist(point_set[s], point_set[index])
        dist_arr[index] = t
        edge_arr[index] = s
        ratio_arr[index] = t 
        if t < temp_dist1 and index in nodes_unfinished :
            temp_dist1 = t
            position = index
    nodes_finished.append(position)
    nodes_unfinished.remove(position)
    result_set.append([s,position,1])
    for index in range(len(point_set)):
        ratio_arr[index] = dist_arr[index] / noise_th
    q_index = 0
    weights = [temp_dist1]
    while True :
        min_ratio = 1.0e14
        for point_i in nodes_unfinished :
            new_node = nodes_finished[-1]
            d = dist(point_set[new_node], point_set[point_i])
            r = d / noise_th
            if r < ratio_arr[point_i] :
                dist_arr[point_i] = d
                ratio_arr[point_i] = r
                edge_arr[point_i] = new_node
            if ratio_arr[point_i] < min_ratio  :
                min_ratio = ratio_arr[point_i]
                position = nodes_unfinished.index(point_i)
                q_index = point_i
        if min_ratio > mean(weights) + std(weights) :
            break
        nodes_finished.append(q_index)
        temp_dist1 = dist_arr[q_index]
        nodes_unfinished.remove(nodes_unfinished[position])
        result_set.append([edge_arr[q_index], q_index, min_ratio])
        weights.append(min_ratio)
        weights.sort()
    return result_set

def smst_od(point_set, sorted_edge):
    least_point = sqrt(len(point_set) / len(point_set[0]))
    # least_point = 6
    noise_th = get_mean_std(sorted_edge)
    labels = [1] * len(point_set)
    window_size =  6
    nodes_finished = []; nodes_unfinished = []
    for i in range(len(point_set)) :
        nodes_unfinished.append(i)
    for index, edge in enumerate(sorted_edge) :
        if edge[0] in nodes_finished or edge[1] in nodes_finished :
            continue
        window = []
        for i in range(window_size) :
            if i + index < len(sorted_edge) :
                window.append(sorted_edge[i+index][2])
        edge_threshold = mean(window)
        if noise_th > edge_threshold :
            s = edge[0]
            temp_mst = update_nodes(point_set, s, noise_th, nodes_finished, nodes_unfinished)
            if len(temp_mst) > least_point :
                for edge in temp_mst:
                    labels[edge[0]] = 0
                    labels[edge[1]] = 0
        else:
            break
    return labels

def prim_mst(point_set):
    result_set = []
    nodes_finished = []; nodes_unfinished = []
    nodes_finished.append(0)
    dist_arr = [0] * len(point_set)
    edge_arr = [-1]* len(point_set)
    temp_dist1 = 1.0e14; position = -1
    for index in range(len(point_set)):
        if index == 0 :
            continue
        t = dist(point_set[0], point_set[index])
        dist_arr[index] = t
        edge_arr[index] = 0
        if t < temp_dist1 :
            temp_dist1 = t
            position = index
    nodes_finished.append(position)
    result_set.append([0, position, temp_dist1])
    for index in range(len(point_set)):
        if index != 0 and index != position :
            nodes_unfinished.append(index)
    q_index = 0
    while len(nodes_unfinished) > 0 :
        temp_dist2 = 1.0e14
        new_node = nodes_finished[-1]
        for point_i in nodes_unfinished :
            d = dist(point_set[new_node], point_set[point_i])
            if d < dist_arr[point_i] : #and r != 0 :
                dist_arr[point_i] = d
                edge_arr[point_i] = new_node
            if dist_arr[point_i] < temp_dist2  :
                temp_dist2 = dist_arr[point_i]
                q_index = point_i
        nodes_finished.append(q_index)
        nodes_unfinished.remove(q_index)
        result_set.append([edge_arr[q_index], q_index,dist_arr[q_index]])
    return result_set, edge_arr, dist_arr

def SMOD(point_set):
    result_set, edge_arr, dist_arr = prim_mst(point_set)
    sorted_edge = sorted(result_set, key = lambda x:x[2])
    preds = smst_od(point_set, sorted_edge)
    return preds

# case4: 0 : id; -1: outlier
def load_arff4(fileName):
	with open(fileName) as fh:
		dataset = np.array(arff.load(fh)['data'])
		point_set = dataset[:,1:-1].astype(np.float)
		labels = dataset[:,-1]
		outlier_num = 0
		for i, l in enumerate(labels):
			if l == 'no' :
				labels[i] = 0
			else:
				labels[i] = 1
				outlier_num += 1
	return point_set, labels.astype(np.int), outlier_num

# case2: -2: id  -1: outlier 
def load_arff2(fileName):
    with open(fileName) as fh:
        dataset = array(arff.load(fh)['data'])
        point_set = dataset[:,:-2].astype(float)
        labels = dataset[:,-1]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no' :
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num

def plt_roc(TPR, FPR, outlier_num, fileName):
    plt.figure(0)
    plt.plot(FPR, TPR, c="red")
    plt.title("ROC-curve on %s"%(fileName))
    # plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    # plt.show()
    plt.savefig("../result/%s.png"%(fileName.split('/')[2].split('.')[0]), dpi=200)
    plt.close(0)

if __name__ == "__main__" :
    # fileName = "../2d_data/data_caiming.dat"
    # point_set = loadData(fileName, float, ',')
    # # # print(len(point_set), len(point))
    # # plt_point(point_set,fileName)
    # clusters = SMOD(point_set)
    # plt_outliers(point_set, clusters, fileName)    

    fileName = "./real_data/SpamBase_norm_39.arff"  # HeartDisease_withoutdupl_norm_44  Parkinson_withoutdupl_norm_75 Pima_withoutdupl_norm_35
    point_set, labels, point_num = load_arff4(fileName)
    # # print(len(point_set), len(point))
    # plt_point(point_set,fileName)
    clusters = SMOD(point_set)
    pre =  precision_score(labels, clusters)
    rec = recall_score(labels, clusters)
    f1_value = f1_score(labels, clusters)
    # FPR, TPR = get_roc(labels, clusters, outlier_num)
    # plt_roc(FPR, TPR, outlier_num, fileName)
    # roc_auc = auc(FPR, TPR)
    print(fileName, "%0.4f"%(pre), "%0.4f"%(rec), "%0.4f"%(f1_value))

    # p = r'../WDBC/'
    # f1 = open("../result/method2_prf_%s.csv"%("othermat"),'w')
    # for root,dirs,files in os.walk(p): 
    #     for name in files:
    #         fileName = os.path.join(p,name)
    #         file_name, file_type = os.path.splitext(name)
    #         point_set, labels, outlier_num = load_arff2(fileName)
    #         # point_set, labels, outlier_num = load_cls1o(fileName)
    #         # point_set = np.array(point_set.tolist()).astype(np.float)
    #         # labels = np.array(labels).astype(int)
    #         # m = loadmat(fileName)
    #         # point_set = m["X"]; labels = m["y"].ravel()
    #         # outliers_fraction = count_nonzero(labels) / len(labels)
    #         # outliers_percentage = round(outliers_fraction * 100, ndigits=4)
    #         # print(file_name, len(point_set), outlier_num, len(point_set[0]))
    #         clusters = SMOD(point_set)
    #         # print(clusters)
    #         pre =  precision_score(labels, clusters)
    #         rec = recall_score(labels, clusters)
    #         f1_value = f1_score(labels, clusters)
    #         # FPR, TPR = get_roc(labels, clusters, outlier_num)
    #         # plt_roc(FPR, TPR, outlier_num, fileName)
    #         # roc_auc = auc(FPR, TPR)
    #         print(name, "%0.4f"%(pre), "%0.4f"%(rec), "%0.4f"%(f1_value))
    # #         f1.write(name + ',' + str("%0.4f,"%(pre)) + str("%0.4f,"%(rec)) + str("%0.4f,"%(f1_value)) + '\n')
    # # f1.close()
    #         # plt_all_roc(labels, point_set, outlier_num, file_name)
