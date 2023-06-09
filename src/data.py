import numpy as np
import pandas as pd

def read_data(file_name, clients=10):

    if file_name == 'customer':
        ds = pd.read_csv('../dataset/customer_data.csv')
        gt = np.array(pd.crosstab(ds['fea_6'], ds['fea_7']).values)
    elif file_name == 'payment_1':
        ds = pd.read_csv('../dataset/payment_data.csv')
        gt = np.array(pd.crosstab(ds['prod_code'], ds['OVD_t1']).values)
    elif file_name == 'payment_2':
        ds = pd.read_csv('../dataset/payment_data.csv')
        gt = np.array(pd.crosstab(ds['prod_code'], ds['OVD_t2']).values)
    elif file_name == 'payment_3':
        ds = pd.read_csv('../dataset/payment_data.csv')
        gt = np.array(pd.crosstab(ds['prod_code'], ds['OVD_t3']).values)
    elif file_name == 'adult':
        ds = pd.read_csv('../dataset/adult.data', header=None)
        gt = np.array(pd.crosstab(ds.iloc[:, 6], ds.iloc[:, 13]).values)
    elif file_name == 'gtsrb_width':
        ds = pd.read_csv('../dataset/gtsrb.csv')
        gt = np.array(pd.crosstab(ds['Width'], ds['ClassId']).values)
    elif file_name == 'gtsrb_height':
        ds = pd.read_csv('../dataset/gtsrb.csv')
        gt = np.array(pd.crosstab(ds['Height'], ds['ClassId']).values)
    elif file_name == 'gtsrb_x1':
        ds = pd.read_csv('../dataset/gtsrb.csv')
        gt = np.array(pd.crosstab(ds['Roi.X1'], ds['ClassId']).values)
    elif file_name == 'gtsrb_y1':
        ds = pd.read_csv('../dataset/gtsrb.csv')
        gt = np.array(pd.crosstab(ds['Roi.Y1'], ds['ClassId']).values)
    elif file_name == 'gtsrb_x2':
        ds = pd.read_csv('../dataset/gtsrb.csv')
        gt = np.array(pd.crosstab(ds['Roi.X2'], ds['ClassId']).values)
    elif file_name == 'gtsrb_y2':
        ds = pd.read_csv('../dataset/gtsrb.csv')
        gt = np.array(pd.crosstab(ds['Roi.Y2'], ds['ClassId']).values)
    elif file_name == 'mushroom_2_4':
        ds = pd.read_csv('../dataset/mushroom.data', header=None)
        gt = np.array(pd.crosstab(ds.iloc[:, 2], ds.iloc[:, 4]).values)
    elif file_name == 'mushroom_8_13':
        ds = pd.read_csv('../dataset/mushroom.data', header=None)
        gt = np.array(pd.crosstab(ds.iloc[:, 8], ds.iloc[:, 13]).values)
    elif file_name == 'mushroom_14_18':
        ds = pd.read_csv('../dataset/mushroom.data', header=None)
        gt = np.array(pd.crosstab(ds.iloc[:, 14], ds.iloc[:, 18]).values)
    elif file_name == 'mushroom_19_21':
        ds = pd.read_csv('../dataset/mushroom.data', header=None)
        gt = np.array(pd.crosstab(ds.iloc[:, 19], ds.iloc[:, 21]).values)
    elif file_name == 'lymphography':
        ds = pd.read_csv('../dataset/lymphography.data', header=None)
        gt = np.array(pd.crosstab(ds.iloc[:, 14], ds.iloc[:, 18]).values)

    lts = []
    for _ in range(clients):
        lts.append(np.zeros_like(gt))
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            idx = np.random.randint(low=0, high=gt[i][j]+1, size=clients-1).tolist()
            idx.append(0)
            idx.append(gt[i][j])
            idx = np.sort(idx)
            for k in range(len(idx)-1):
                lts[k][i][j] = idx[k+1]-idx[k]
    return lts, gt
