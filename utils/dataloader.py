import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
import torchvision.datasets as ds
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def normalize(image):
    """
    Normalize data between -1 and 1.
    """
    return (
        2
        * torch.div(
            torch.sub(image, torch.min(image)), torch.max(image) - torch.min(image)
        )
        - 1
    )


class DataSet(Dataset):
    def __init__(self, data, labels, transform=Compose([ToTensor(), normalize])):
        super(DataSet, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)


def encode_to_one_hot_vec(df, name):
    """
    Encode text values to one hot vector.
    """
    dummies = pd.get_dummies(df.loc[:, name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def load_kdd(csv_path):
    """
    Load KDD Cup dataset.
    """
    col_names = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
    ]

    csv = pd.read_csv(csv_path, header=None, names=col_names)

    categorical_col = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]

    for name in categorical_col:
        encode_to_one_hot_vec(csv, name)

    labels = csv["label"].copy()

    # As in the Paper said: "Due to the high proportion of outliers
    # in the KDD dataset, “normal” data are treated as anomalies [...].
    # labels[labels != 'normal.'] = 0.0  # zero means normal

    binary_labels = np.zeros(labels.shape[0])  # zero means normal
    binary_labels[labels == "normal."] = 1.0  # one means anomalous

    csv["label"] = binary_labels

    # Rearrange position of column label
    new_columns = csv.columns.to_list()
    new_columns.remove("label")
    new_columns.append("label")

    csv = csv.reindex(columns=new_columns)

    return csv


def Dataloader(dataset, normal_class=0, batch_size=32):
    """
    Reads either CIFAR-10 or KDD Cup 99 dataset and returns data in batches.
    """
    if dataset == "CIFAR10":
        print(f"Load image data CIFAR-10 of class {normal_class}...")
        trainset = ds.CIFAR10(
            root=f"{os.getcwd()}/data/train", train=True, download=True
        )

        train_data = trainset.data
        labels = np.array(trainset.targets)

        # As in the Paper said: "[...] treating images from one class as normal and
        # considering images from the remaining 9 as anomalous.
        # [...] discard anomalous samples from both training and validation sets"
        normal_data = train_data[labels == normal_class]
        normal_labels = labels[labels == normal_class]

        print(
            f"After discarding anomalous classes (except normal class {normal_class}), "
            f"{normal_data.shape[0]} of {train_data.shape[0]} "
            f"images ({normal_data.shape[0]/train_data.shape[0]*100}%) remain for training."
        )

        train_size = int(len(normal_data) * 0.75)

        x_train = normal_data[:train_size]
        y_train = normal_labels[:train_size]

        x_valid = normal_data[train_size:]
        y_valid = normal_data[train_size:]

        testset = ds.CIFAR10(
            root=f"{os.getcwd()}/data/test", train=False, download=True
        )

        x_test = testset.data
        labels = np.array(testset.targets)

        y_test = np.ones(labels.shape[0])  # one means anomalous
        y_test[labels == normal_class] = 0.0  # zero means normal

        transform = Compose([ToTensor(), normalize])

    elif dataset == "KDDCup":
        print("Load tabular data KDDCup99...")

        train_csv_path = f"{os.getcwd()}/data/train/kddcup99/kddcup.data_10_percent.gz"
        csv_df = load_kdd(train_csv_path)

        # The test data is not labeled. Therefore, we have to split the train set on our own.
        train_df = csv_df.sample(frac=0.8, random_state=42)
        test_df = csv_df.loc[~csv_df.index.isin(train_df.index)]

        x, y = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
        x_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

        # 25 % of the train data should be used for validation
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.25, random_state=42
        )

        # Exclude as anomalous defined samples
        x_train = x_train[y_train != 1]
        y_train = y_train[y_train != 1]
        x_valid = x_valid[y_valid != 1]
        y_valid = y_valid[y_valid != 1]

        x_train = torch.Tensor(x_train.astype(np.float32))
        x_test = torch.Tensor(x_test.astype(np.float32))
        x_valid = torch.Tensor(x_valid.astype(np.float32))
        y_train = torch.Tensor(y_train.astype(np.float32))
        y_test = torch.Tensor(y_test.astype(np.float32))
        y_valid = torch.Tensor(y_valid.astype(np.float32))

        transform = None

    else:
        raise NameError("Requested dataset is not available.")

    # Calculate number of anomalous samples for testing
    num_anomalous_sample = np.where(y_test == 1.0)[0].shape[0]

    print(
        f"Proportion of as anomalous defined samples in test set: "
        f"{np.round(num_anomalous_sample / y_test.shape[0] * 100, decimals=4)}%"
        f" (In Total: {num_anomalous_sample} of {y_test.shape[0]} samples)"
    )

    train_ds = DataSet(x_train, y_train, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    valid_ds = DataSet(x_valid, y_valid, transform=transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=True
    )

    test_ds = DataSet(x_test, y_test, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, valid_loader, test_loader, num_anomalous_sample
