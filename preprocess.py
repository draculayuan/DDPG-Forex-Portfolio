import numpy as np
import pandas as pd


def split_by_hour(df):
    S = pd.to_datetime(df.iloc[:,0])
    out = [g.reset_index(drop=True) for i, g in df.groupby([(S - S[0]).astype('timedelta64[h]')])]
    new_df = []
    for i in range(len(out)):
        temp = out[i].iloc[:,1:5].values
        open_bid = temp[0,0]
        high_bid = np.max(temp[:,1])
        low_bid = np.min(temp[:,2])
        close_bid = temp[-1,3]
        new_df.append([open_bid, high_bid, low_bid, close_bid])        
    return np.array(new_df)

def get_data():
    data1_05 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdhkd/DAT_XLSX_USDHKD_M1_201805.xlsx")
    data1_06 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdhkd/DAT_XLSX_USDHKD_M1_201806.xlsx")
    data1_07 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdhkd/DAT_XLSX_USDHKD_M1_201807.xlsx")
    #data1 = data1.iloc[:, 1:5]
    #data1 = data1.values

    data2_05 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdsgd/DAT_XLSX_USDSGD_M1_201805.xlsx")
    data2_06 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdsgd/DAT_XLSX_USDSGD_M1_201806.xlsx")
    data2_07 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdsgd/DAT_XLSX_USDSGD_M1_201807.xlsx")
    #data2 = data2.iloc[:, 1:5]
    #data2 = data2.values

    data3_05 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usddkk/DAT_XLSX_USDDKK_M1_201805.xlsx")
    data3_06 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usddkk/DAT_XLSX_USDDKK_M1_201806.xlsx")
    data3_07 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usddkk/DAT_XLSX_USDDKK_M1_201807.xlsx")

    data4_05 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdsek/DAT_XLSX_USDSEK_M1_201805.xlsx")
    data4_06 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdsek/DAT_XLSX_USDSEK_M1_201806.xlsx")
    data4_07 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/train/usdsek/DAT_XLSX_USDSEK_M1_201807.xlsx")
    '''
    data3_05 = pd.read_excel("/raid0/students/student01/git_folder/DDPG-Keras-Forex/data/train/eurnok/DAT_XLSX_EURNOK_M1_201805.xlsx")
    data3_06 = pd.read_excel("/raid0/students/student01/git_folder/DDPG-Keras-Forex/data/train/eurnok/DAT_XLSX_EURNOK_M1_201806.xlsx")
    data3_07 = pd.read_excel("/raid0/students/student01/git_folder/DDPG-Keras-Forex/data/train/eurnok/DAT_XLSX_EURNOK_M1_201807.xlsx")

    data4_05 = pd.read_excel("/raid0/students/student01/git_folder/DDPG-Keras-Forex/data/train/eurnzd/DAT_XLSX_EURNZD_M1_201805.xlsx")
    data4_06 = pd.read_excel("/raid0/students/student01/git_folder/DDPG-Keras-Forex/data/train/eurnzd/DAT_XLSX_EURNZD_M1_201806.xlsx")
    data4_07 = pd.read_excel("/raid0/students/student01/git_folder/DDPG-Keras-Forex/data/train/eurnzd/DAT_XLSX_EURNZD_M1_201807.xlsx")
    '''
    new_data1_05 = split_by_hour(data1_05)
    new_data1_06 = split_by_hour(data1_06)
    new_data1_07 = split_by_hour(data1_07)
    new_data2_05 = split_by_hour(data2_05)
    new_data2_06 = split_by_hour(data2_06)
    new_data2_07 = split_by_hour(data2_07)
    new_data3_05 = split_by_hour(data3_05)
    new_data3_06 = split_by_hour(data3_06)
    new_data3_07 = split_by_hour(data3_07)
    new_data4_05 = split_by_hour(data4_05)
    new_data4_06 = split_by_hour(data4_06)
    new_data4_07 = split_by_hour(data4_07)
    
    test1 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/test/usdhkd/DAT_XLSX_USDHKD_M1_201809.xlsx")
    test2 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/test/usdsgd/DAT_XLSX_USDSGD_M1_201809.xlsx")
    test3 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/test/usddkk/DAT_XLSX_USDDKK_M1_201809.xlsx")
    test4 = pd.read_excel("/Users/liuyuan/Desktop/git/DDPG-Keras-Torcs/data/test/usdsek/DAT_XLSX_USDSEK_M1_201809.xlsx")
    test1 = split_by_hour(test1)
    test2 = split_by_hour(test2)
    test3 = split_by_hour(test3)
    test4 = split_by_hour(test4)

    train1 = np.vstack([new_data1_05[:490,:], new_data1_06[:490,:], new_data1_07[:490,:]]) 
    train2 = np.vstack([new_data2_05[:490,:], new_data2_06[:490,:], new_data2_07[:490,:]]) 
    train3 = np.vstack([new_data3_05[:490,:], new_data3_06[:490,:], new_data3_07[:490,:]]) 
    train4 = np.vstack([new_data4_05[:490,:], new_data4_06[:490,:], new_data4_07[:490,:]]) 
    #data = np.hstack([new_data1,new_data2])
    train = np.hstack([train1,train2,train3,train4])

    test1 = test1[:470, :]
    test2 = test2[:470, :]
    test3 = test3[:470, :]
    test4 = test4[:470, :]
    test = np.hstack([test1, test2, test3, test4])
    
    return train, test