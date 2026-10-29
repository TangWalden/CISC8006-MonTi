import json

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from modules.data_handler import DataHandler
from utils.utils import get_n_ids
from course.improvements.policies import group_sampling_weights


class RecordedLoader:
    def __init__(self, loader, path, sizes):
        self.loader, self.path, self.sizes = loader, path, sizes

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        groups, complete = [], False
        try:
            for batch in self.loader:
                groups.extend(batch[0].tolist())
                yield batch
            complete = True
        finally:
            record = dict(complete=complete, groups=groups, group_updates=len(groups),
                          target_occurrences=sum(self.sizes[group] for group in groups))
            self.path.write_text(json.dumps(record) + '\n')


class SamplingDataHandler(DataHandler):
    def __init__(self, config, evidence, mode):
        self.sampling_evidence, self.sampling_mode, self.sampling_epoch = evidence, mode, 0
        super().__init__(config)

    def get_train_loader(self):
        self.sampling_epoch += 1
        ids = torch.LongTensor(self.get_train_set())
        sizes = {int(group): len(get_n_ids(self.target_assignment, group)) for group in ids}
        if self.sampling_mode == 'baseline':
            loader = super().get_train_loader()
        else:
            weights = group_sampling_weights([sizes[int(group)] for group in ids], self.sampling_mode)
            dataset = TensorDataset(ids, torch.LongTensor(self.node_budgets[ids]), torch.LongTensor(self.edge_budgets[ids]))
            sampler = WeightedRandomSampler(weights, num_samples=len(ids), replacement=True)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0)
        return RecordedLoader(loader, self.sampling_evidence / ('sampling_epoch_%03d.json' % self.sampling_epoch), sizes)
