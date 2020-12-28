from .dss_parameter import DSSParameter
from collections.abc import MutableMapping
from typing import Any


class DkuConfig(MutableMapping):
    """Mapping structure containing DSSParameter objects. It behaves as a dict with the following differences:
        - You can access elements with a dot structure (Example: dku_config.param1 or dku_config["param1"])
        - You can set an element with a dot structure (Example: dku_config.param1 = 123)
        - All objects stored are converted in DSSParameter
        - Accessing an element returns the value of the object DSSParameter

    Attributes:
        config(dict): Dict storing the DSSParameters
    """
    def __init__(self, **kwargs):
        """Initialization method for the DkuConfig class

        Args:
            **kwargs: DSSParameters. Each key will be set as the parameter name and the values must be of type
                dict. These dicts must contain at least an attribute "value". For other attributes, see
                DSSParameter help.
        """
        object.__setattr__(self, 'config', {})
        if kwargs:
            for k, v in kwargs.items():
                if 'value' not in v:
                    raise ValueError('Each init kwargs must have a "value" field.')
                val = v.pop('value')
                self.add_param(name=k, value=val, **v)

    def add_param(self, name: str, value: Any, **kwargs):
        """Add a new DSSParameter to the config

        Args:
            name(str): The name of the parameter
            value(anytype): The value of the parameter
            **kwargs: Other arguments. See DSSParameter help.
        """
        self.config[name] = DSSParameter(name=name, value=value, **kwargs)

    def get_param(self, name: str) -> DSSParameter:
        """Returns the DSSParameter of given name

        Args:
            name(str): Name of object to return

        Returns:
            DSSParameter: Parameter of given name
        """
        return self.config.get(name)

    def __delitem__(self, item):
        del self.config[item]

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        if item in self.config:
            return self.config.get(item).value
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        self.add_param(name=key, value=value)

    def __iter__(self):
        return iter(self.config)

    def __len__(self):
        return len(self.config)

    def __repr__(self):
        return self.config.__repr__()

    def __str__(self):
        return self.config.__str__()
