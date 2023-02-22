from torch.utils.data.sampler import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class OrderedSampler(Sampler[int]):
    r"""Samples elements in the order specified.
    Args:
        order: a sequence of indices
    """
    def __init__(self, order: List[int]) -> None:
        self.order = order


    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order.length)