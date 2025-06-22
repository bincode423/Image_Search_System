import collections
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from PIL import Image

class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m) # 4
        self.labels_to_indices = self.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys()) # 100
        self.length_of_single_pass = self.m_per_class * len(self.labels) # 400
        self.list_size = length_before_new_iter # 30000

    def get_labels_to_indices(self, labels):
            """
            Creates labels_to_indices, which is a dictionary mapping each label
            to a numpy array of indices that will be used to index into self.dataset
            """
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            labels_to_indices = collections.defaultdict(list)
            for i, label in enumerate(labels):
                labels_to_indices[label].append(i)
            for k, v in labels_to_indices.items():
                labels_to_indices[k] = np.array(v, dtype=int)
            return labels_to_indices

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            curr_label_set = self.labels
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = np.random.choice(t, size=self.m_per_class)
                i += self.m_per_class
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass
        )
        return self.list_size // divisor