# sl_core/client.py
import torch
import torch.nn as nn

class Client:
    def __init__(self, cid, head, tail, loader, lr_head=0.01, lr_tail=0.01, device='cpu'):
        self.id = cid
        self.device = device
        self.head = head.to(device)
        self.tail = tail.to(device)
        self.loader = loader

        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=lr_head, momentum=0.9)
        self.opt_tail = torch.optim.SGD(self.tail.parameters(), lr=lr_tail, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    # def run_batches(self, server, loader=None, max_steps=None, attack_hook=None):
    #     self.head.train()
    #     self.tail.train()
    #     server.model.train() # Backbone

    #     steps = 0
    #     total_loss = 0.0
        
    #     # [修改]：初始化特征累加器，用于第一阶段的 Feature-DDifs
    #     total_features = None 

    #     use_loader = loader if loader is not None else self.loader

    #     for x, y in use_loader:
    #         x = x.to(self.device)
    #         y = y.to(self.device)

    #         if attack_hook is not None:
    #             x, y = attack_hook(x, y, self.id)

    #         a = self.head(x)
            
    #         # [修改]：旁路拦截前向特征，进行全局平均池化 (压缩 Batch, H, W 维度)
    #         with torch.no_grad():
    #             batch_features = a.mean(dim=(0, 2, 3))
    #             if total_features is None:
    #                 total_features = batch_features.clone()
    #             else:
    #                 total_features += batch_features

    #         a.retain_grad()

    #         # a_server = a.detach().clone().to(server.device).requires_grad_(True)
    #         # b_server = server.model(a_server)
    #         #0604
    #         a_server = a.detach().clone().to(server.device).requires_grad_(True)
            
    #         # ================== 【新增：在线动态通道掩码防御】 ==================
    #         # 检查服务端是否已经拥有了本轮的健康通道基准
    #         if hasattr(server, 'global_anchor') and server.global_anchor is not None:
    #             with torch.no_grad():
    #                 # 计算当前输入在 32 个通道上的平均活跃度
    #                 current_means = a_server.mean(dim=(0, 2, 3))
    #                 # 计算与健康基准的比例
    #                 ddifs_ratio = current_means / (server.global_anchor + 1e-8)
                    
    #                 # 设定阈值：大于 10.0 倍的被视为后门触发器引起的绝对异常
    #                 BACKDOOR_THRESHOLD = 10.0
    #                 mask = (ddifs_ratio < BACKDOOR_THRESHOLD).float()
                    
    #                 # 打印被绞杀的通道
    #                 if (mask == 0).any():
    #                     bad_channels = torch.where(mask == 0)[0].tolist()
    #                     print(f"    [Active Masking] Client {self.id} 触发器异常激增！强行关闭通道: {bad_channels}")

    #             # 将掩码应用到高维特征上，彻底抹除触发器语义
    #             a_server_sanitized = a_server * mask.view(1, -1, 1, 1)
    #         else:
    #             a_server_sanitized = a_server
    #         # ====================================================================
            
    #         b_server = server.model(a_server_sanitized)


            
    #         b_tail = b_server.detach().to(self.device).requires_grad_(True)
    #         logits = self.tail(b_tail)
    #         loss = self.criterion(logits, y)
            
    #         self.opt_head.zero_grad(set_to_none=True)
    #         self.opt_tail.zero_grad(set_to_none=True)
    #         loss.backward()

    #         g_b = b_tail.grad.detach()
            
    #         server.opt.zero_grad(set_to_none=True)
    #         b_server.backward(g_b.to(server.device))
    #         server.opt.step()

    #         g_a = a_server.grad.detach()
            
    #         g_a_client = g_a.to(self.device)
    #         torch.autograd.backward(a, g_a_client)
    #         self.opt_tail.step()
    #         self.opt_head.step()

    #         total_loss += loss.item()
            
    #         steps += 1
    #         if max_steps is not None and steps >= max_steps:
    #             break

    #     # [修改]：计算平均通道特征并返回，供第一阶段分析
    #     avg_features = total_features / max(1, steps)
        
    #     return {
    #         'loss': total_loss / max(1, steps), 
    #         'steps': steps,
    #         'phase1_features': avg_features.cpu()
    #     }

    def run_batches(self, server, loader=None, max_steps=None, attack_hook=None):
        self.head.train()
        self.tail.train()
        server.model.train() # Backbone

        steps = 0
        total_loss = 0.0
        
        # [修改]：初始化特征累加器与梯度累加器
        total_features = None
        total_feature_map = None
        total_grad_norm = 0.0  # <--- 【新增】初始化

        use_loader = loader if loader is not None else self.loader

        for x, y in use_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if attack_hook is not None:
                x, y = attack_hook(x, y, self.id)

            a = self.head(x)
            
            with torch.no_grad():
                batch_features = a.mean(dim=(0, 2, 3))
                if total_features is None:
                    total_features = batch_features.clone()
                else:
                    total_features += batch_features

                batch_feature_map = a.detach().mean(dim=0)
                if total_feature_map is None:
                    total_feature_map = batch_feature_map.clone()
                else:
                    total_feature_map += batch_feature_map

            a.retain_grad()

            a_server = a.detach().clone().to(server.device).requires_grad_(True)
            
            # === 在线动态通道掩码防御 ===
            if hasattr(server, 'global_anchor') and server.global_anchor is not None:
                with torch.no_grad():
                    current_means = a_server.mean(dim=(0, 2, 3))
                    ddifs_ratio = current_means / (server.global_anchor + 1e-8)
                    BACKDOOR_THRESHOLD = 10.0
                    mask = (ddifs_ratio < BACKDOOR_THRESHOLD).float()
                    if (mask == 0).any():
                        bad_channels = torch.where(mask == 0)[0].tolist()
                        print(f"    [Active Masking] Client {self.id} 强行关闭通道: {bad_channels}")
                a_server_sanitized = a_server * mask.view(1, -1, 1, 1)
            else:
                a_server_sanitized = a_server
            # ============================
            
            b_server = server.model(a_server_sanitized)
            
            b_tail = b_server.detach().to(self.device).requires_grad_(True)
            logits = self.tail(b_tail)
            loss = self.criterion(logits, y)
            
            self.opt_head.zero_grad(set_to_none=True)
            self.opt_tail.zero_grad(set_to_none=True)
            loss.backward()

            g_b = b_tail.grad.detach()
            
            server.opt.zero_grad(set_to_none=True)
            b_server.backward(g_b.to(server.device))
            server.opt.step()

            g_a = a_server.grad.detach()
            
            # <--- 【新增】计算并累加这一步的梯度范数
            grad_norm = torch.norm(g_a, p=2).item()
            total_grad_norm += grad_norm 
            
            g_a_client = g_a.to(self.device)
            torch.autograd.backward(a, g_a_client)
            self.opt_tail.step()
            self.opt_head.step()

            total_loss += loss.item()
            
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        avg_features = total_features / max(1, steps)
        avg_feature_map = total_feature_map / max(1, steps)
        
        return {
            'loss': total_loss / max(1, steps), 
            'steps': steps,
            'phase1_features': avg_features.cpu(),
            'phase1_feature_map': avg_feature_map.cpu(),
            'phase1_norm': total_grad_norm / max(1, steps) # <--- 【新增】返回 Phase 1 的梯度范数
        }