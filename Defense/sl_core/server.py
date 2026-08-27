# sl_core/server.py

import torch



class ServerBackbone:

    def __init__(self, backbone, lr=0.01, device='cuda'):

        # 兼容 device 为字符串或 torch.device 对象，强制对齐设备

        if isinstance(device, torch.device):

            self.device = device

        else:

            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

            

        self.model = backbone.to(self.device)

        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)



    @torch.no_grad()

    def forward_only(self, a_cpu):

        # 验证时的前向传播

        a = a_cpu.to(self.device)

        b = self.model(a)

        return b.detach().cpu().float()