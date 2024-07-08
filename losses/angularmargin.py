import torch
import torch.nn.functional as F


class AngularMarginLoss(torch.nn.Module):
    def __init__(self, margin=0.5, scale=30.0):
        super(AngularMarginLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(torch.pi) - margin)
        self.mm = torch.sin(torch.tensor(torch.pi) - margin) * margin

    def forward(self, logits, labels):
        cos_theta = logits
        cos_theta = cos_theta.clamp(-1, 1)  # Ensure cos_theta is in the range [-1, 1]
        target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), labels].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m

        cond_v = target_logit - self.th
        keep_val = (cond_v <= 0).float()
        cos_theta_m = cos_theta_m * (1 - keep_val) + (target_logit - self.mm) * keep_val

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        output *= self.scale

        loss = F.cross_entropy(output, labels)
        return loss

# 예시 사용법
if __name__ == "__main__":
    batch_size = 16
    num_classes = 10
    features_dim = 512

    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    criterion = AngularMarginLoss(margin=0.5, scale=30.0)
    loss = criterion(logits, labels)
    print(loss)