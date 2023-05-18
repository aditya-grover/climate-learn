# Standard library
from __future__ import annotations
from abc import ABC
import copy
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
if TYPE_CHECKING:
    from ..climate_dataset import ClimateDataset


class ClimateDatasetArgs(ABC):
    r"""Data class that stores necessary arguments to initialize the abstract :py:class:`~climate_learn.data.climate_dataset.ClimateDataset`"""
    _data_class: Union[Callable[..., ClimateDataset], str] = "ClimateDataset"

    def __init__(
        self,
        variables: Sequence[str],
        constants: Sequence[str] = [],
        name: str = "climate_dataset",
    ) -> None:
        r"""
        .. highlight:: python

        :param variables: List of variables for which the Climate Dataset should support queries
        :type variables: Sequence[str]
        :param constants: List of constants for which the Climate Dataset should support queries
        :type constants: Sequence[str]
        :param split: The name of the split correpsonding to which the Climate Dataset would load
            the data for. Supported name for splits include `['train', 'val', 'test']`
        :type split: str
        """
        self.variables: Sequence[str] = variables
        self.constants: Sequence[str] = constants
        self.name: str = name
        ClimateDatasetArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any] = {}) -> ClimateDatasetArgs:
        r"""
        Returns a near identical copy of the current instance of the class.
            Useful for cases when need to create an almost identical copy of the
            current instance of the class but with few changes. The changes are
            passed by args, which is a dict. The keys of this dict should be the
            attribute name and value should be the new value for the corresponding
            attribute.

        .. highlight:: python

        :param args: A dict whose keys are the attribute names and values are the
            new value for the corresponding attribute.
        :type args: Dict[str, Any]
        """
        new_instance: ClimateDatasetArgs = copy.deepcopy(self)
        for arg in args:
            if hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        ClimateDatasetArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        r"""
        Checks whether the attributes of the current instance of class hold legal values.
        """
        pass
