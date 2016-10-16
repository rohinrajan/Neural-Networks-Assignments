import numpy as np

def read_csv_as_matrix(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    return data


def normalize_dataset(dataset):
    max_value_price = np.max(dataset.T[0])
    min_value_price = np.min(dataset.T[0])
    max_value_volume = np.max(dataset.T[1])
    min_value_volume = np.min(dataset.T[1])
    new_dataset = dataset.T
    new_dataset[0] = (new_dataset[0] - min_value_price) / (max_value_price - min_value_price)
    new_dataset[1] = (new_dataset[1] - min_value_volume) / (max_value_volume - min_value_volume)
    dataset = new_dataset.T
    return dataset

if __name__ == "__main__":
    dataset1 = read_csv_as_matrix("stock_data.csv")
    print dataset1.shape
    print dataset1[0]
    dataset1 = normalize_dataset(dataset1)
    print dataset1.shape
    print dataset1[0]
    # print dataset1[0].reshape(2,1)
    # print dataset1[0]
