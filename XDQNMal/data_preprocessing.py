import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def remove_all_zero_rows(file_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)

    # 创建一个布尔索引，用于标记除了第1和第2列之外，是否整行全为0
    mask = (df.iloc[:, 2:] == 0).all(axis=1)

    # 过滤掉整行全为0的行
    df_filtered = df[~mask]

    # 将过滤后的数据保存到新的CSV文件
    df_filtered.to_csv(output_path, index=False, header=None)


def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data = data.drop(data.columns[[2]], axis=1)
    data = data.drop(data.columns[[1]], axis=1)
    # data = data.iloc[2247:]
    X = data.drop(data.columns[[0]], axis=1)
    y = data[data.columns[0]]
    X = X.values
    y = y.values

    return X, y


# def normalize_data(X_train, X_test):
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     return X_train_scaled, X_test_scaled

def normalize_data(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X


def standard_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def replace_excess_with_nan(arr, k):
    # 确保输入是numpy数组
    arr = np.array(arr)
    rows, cols = arr.shape

    for i in range(rows):
        counts = {}  # 用于跟踪每个数字出现的次数
        # 计算中间点，向两边遍历
        mid = cols // 2
        for j in range(cols):
            # 选择从中间向两边的索引
            if j % 2 == 0:
                idx = mid + j // 2
            else:
                idx = mid - (j + 1) // 2

            num = arr[i, idx]
            if num in counts:
                counts[num] += 1
                if counts[num] > k:
                    arr[i, idx] = np.nan  # 超过k次，用nan替换
            else:
                counts[num] = 1  # 首次出现
    return arr


def shift_values(arr):
    rows, cols = arr.shape
    for row in range(rows):
        filtered_row = arr[row, ~np.isnan(arr[row, :])]
        nan_count = cols - len(filtered_row)
        arr[row, :] = np.concatenate([filtered_row, [np.nan] * nan_count])
    return arr
