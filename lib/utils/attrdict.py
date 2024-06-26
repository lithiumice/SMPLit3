class attr_dict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                "update",
                "pop",
            ):
                setattr(self, k, getattr(self, k))

    def to_dict(self):
        ret_dict = {}
        for k, v in self.__dict__.items():
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                "update",
                "pop",
                "to_dict",
            ):
                ret_dict[k] = v
        return ret_dict

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(attr_dict, self).__setattr__(name, value)
        super(attr_dict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(attr_dict, self).pop(k, d)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
