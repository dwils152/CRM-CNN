from torch.utils.data import DataLoader

def get_data_loader(data, batch_size, sampler=None):
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        persistent_workers=True,
        pin_memory=True,
        sampler=sampler,
    )
    return loader
