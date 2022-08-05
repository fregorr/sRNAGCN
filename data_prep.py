import os
import os.path as osp
import pickle as rick
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, download_url



class MyDataset(Dataset):
    def __init__(self, root, input_list_size, content_raw, path_to_processed, transform=None, pre_transform=None):
        # root = Where the Dataset should be stored. This Folder is
        # split into raw_dir (where the raw data should be or is downloaded to)
        # and processed_dir (where the processed data is saved).
        # I use a path to the raw data iin the process function and don't
        # use the raw_dir.
        self.input_list_size = input_list_size
        self.content_raw = content_raw
        self.path_to_processed = path_to_processed

        super().__init__(root, transform, pre_transform)
        # It can be that it should be:
        # super(MyDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.content_raw

    @property
    def processed_file_names(self):
        content_processed = os.listdir(self.path_to_processed)
        if len(content_processed) > 10:
            file_names = content_processed
            try:
                file_names.remove("pre_filter.pt")
                file_names.remove("pre_transform.pt")
            except:
                print("No prefilter and pretransform files")
        else:
            file_names = ["data_1.pt", "data_2.pt"]
        return file_names

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # print(raw_path)   # debugging
            pickle_in = open(raw_path, "rb")
            for step in range(0, self.input_list_size):
                try:
                    entry = rick.load(pickle_in)
                    features = torch.tensor(entry[0], dtype=torch.float)
                    edge_index = torch.tensor(entry[1], dtype=torch.long)
                    edge_index = edge_index.t().contiguous()
                    attr = torch.tensor(entry[2])
                    edge_attr = attr.float()  # necessary, because edge_attr must be in different
                    # shape, than edge_weight
                    label = entry[3]
                    #label_tensor = torch.tensor([label], dtype=torch.float)  # float for Regression
                    label_tensor = torch.tensor([label], dtype=torch.long)  # Long for Classification
                    intarna_energy = entry[4]
                    intarna_energy_tensor = torch.tensor([intarna_energy])
                    #if entry[5] == "train":
                    #    split = torch.tensor([0])
                    #elif entry[5] == "val":
                    #    split = torch.tensor([1])
                    #rna_ids = torch.tensor(entry[6])
                    data_entry = Data(x=features,
                                      edge_index=edge_index,
                                      edge_attr=edge_attr,
                                      y=label_tensor,
                                      intarna_energy=intarna_energy_tensor
                                      )
                    torch.save(data_entry,
                               osp.join(self.processed_dir,
                                        f'data_{idx}.pt'))
                    if idx % 100 == 0:
                        print(idx)
                    idx = idx + 1
                except:
                    break
            pickle_in.close()
        idx = idx - 1
        print(idx)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   f'data_{idx}.pt'))
        return data
