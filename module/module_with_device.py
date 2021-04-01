from torch import nn


class ModuleWithDevice(nn.Module):
    """A torch module with device."""

    def __init__(self):
        super(ModuleWithDevice, self).__init__()
        self.device = None

    def set_device(self, device):
        self.device = device
        self.to(device)

    def assure_device(self, *args):
        assert self.device, 'Device of module should be set first'
        res = []
        for data in args:
            if not hasattr(data, 'device') or data.device != self.device:
                res.append(data.to(self.device))
            else:
                res.append(data)
        if len(res) == 1:
            return res[0]
        return res
