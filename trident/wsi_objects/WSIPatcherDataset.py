from torch.utils.data import Dataset


def collate_pil_batch(items):
    """DataLoader collate for a ``WSIPatcher`` (``pil=True``): keep the batch of ``(PIL tile, x, y)``
    as parallel ``(tiles, xys)`` lists (default collate can't stack PIL images). Module-level so it
    stays picklable under the spawn context."""
    return [it[0] for it in items], [(it[1], it[2]) for it in items]


class WSIPatcherDataset(Dataset):
    """ Dataset from a WSI patcher to directly read tiles on a slide  """
    
    def __init__(self, patcher, transform):
        self.patcher = patcher
        self.transform = transform
                              
    def __len__(self):
        return len(self.patcher)
    
    def __getitem__(self, index):
        tile, x, y = self.patcher[index]

        if self.transform:
            tile = self.transform(tile)

        return tile, (x, y)
