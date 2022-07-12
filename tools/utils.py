import torch
from torch import nn


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='robertaembeddings.word_embeddings_layer.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 这里加入fgm attack来判断是否进行对抗训练了
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='robertaembeddings.word_embeddings_layer.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 这里加入fgm restore判断是否恢复参数了
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        # nn.init.normal_(self.weight)
        # nn.init.normal_(self.bias)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs


class ProjectionMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size
        hidden_dim = hidden_size * 2
        out_dim = hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU()]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class EntityEmbeddings(nn.Module):
    def __init__(self, config, is_pos_embedding=False):
        super().__init__()
        self.entity_embeddings = nn.Embedding(config.entity_queries_num, config.hidden_size)
        self.use_common_embedding = config.use_entity_common_embedding and not is_pos_embedding
        if self.use_common_embedding:
            self.entity_common_embedding = nn.Embedding(1, config.hidden_size)
            # self.entity_common_embedding = nn.Parameter(torch.rand(config.hidden_size))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        embeddings = self.entity_embeddings(input_ids)
        if self.use_common_embedding:
            embeddings = embeddings + self.entity_common_embedding(torch.tensor(0, device=embeddings.device))
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PGD():
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
