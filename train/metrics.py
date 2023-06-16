class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v,tuple):
                self.meters[k].update(*v)
            else:
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        print_str = []
        for name, meter in self.meters.items():
            print_str.append(f"{name}: {str(meter.compute().item())}")
        return self.delimiter.join(print_str)

    def compute(self,meters):
        if isinstance(meters,str):
            return self.meters[meters].compute().item()
        return tuple((self.meters[meter].compute().item() for meter in meters))

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def reset(self):
        for meter in self.meters:
            self.meters[meter].reset()