from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
import torch
from .vpt import PromptViT
import os
import logging
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import math

logger = logging.getLogger(__name__)



@ADAPTATION_REGISTRY.register()
class DPCore(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg

        self.lamda = 1.0
        self.temp_tau = cfg.DPCORE.TEMP_TAU
        self.ema_alpha = cfg.DPCORE.EMA_ALPHA
        self.thr_rho = cfg.DPCORE.THR_RHO
        self.E_ID = 1
        self.E_OOD = cfg.DPCORE.E_OOD

        self.max_num_prompts = cfg.DPCORE.MAX_PROTOTYPES
        
        self.model = config_model(model, cfg.DPCORE.NUM_PROMPTS)
        self.optimizer = torch.optim.AdamW([self.model.prompts], 
                                           lr=cfg.OPTIM.LR, 
                                           betas=(cfg.OPTIM.BETA, 0.999),
                                           weight_decay=cfg.OPTIM.WD)
        
        self.get_number_trainable_params([self.model.prompts])
        
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.coreset = []

        self.obtain_src_stat(cfg.DATA_DIR, cfg.DPCORE.SRC_NUM_SAMPLES)

    def get_number_trainable_params(self, params):
        if isinstance(params, list):
            trainable = sum(p.numel() for p in params) if len(params) > 0 else 0
            
        elif isinstance(params, dict):
            trainable = []
            for _, param in params.items():
                if len(param) > 0:
                    trainable.append(sum(p.numel() for p in param))
            trainable = sum(trainable)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% ")
        return trainable, total
    
    def _update_coreset(self, weights, batch_mean, batch_std):
        """Update overall test statistics, Eqn. (9)"""
        updated_prompts = self.model.prompts.clone().detach().cpu()
        for p_idx in range(len(self.coreset)):
            self.coreset[p_idx][0] += self.ema_alpha * weights[p_idx] * (batch_mean - self.coreset[p_idx][0])
            self.coreset[p_idx][1] += self.ema_alpha * weights[p_idx] * torch.clamp(batch_std - self.coreset[p_idx][1], min=0.0)
            self.coreset[p_idx][2] += self.ema_alpha * weights[p_idx] * (updated_prompts - self.coreset[p_idx][2])


    @torch.no_grad()
    def _eval_coreset(self, x):
        """Evaluate the coreset on a batch of samples."""
        
        loss, batch_mean, batch_std = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=False)
        is_ID = False
        weights = None
        weighted_prompts = None
        
        if self.coreset:
            weights = calculate_weights(self.coreset, batch_mean, batch_std, self.lamda, self.temp_tau)
            weighted_prompts = torch.stack([w * p[2] for w, p in zip(weights, self.coreset)], dim=0).sum(dim=0)
            assert weighted_prompts.shape == self.model.prompts.shape, f'{weighted_prompts.shape} != {self.model.prompts.shape}'
            self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
            self.model.prompts.requires_grad_(False)
            
            loss_new, _, _ = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=True)
            if loss_new < loss * self.thr_rho:
                self.model.prompts.requires_grad_(True)
                self.optimizer = torch.optim.AdamW([self.model.prompts], lr=self.cfg.OPTIM.LR)
                is_ID = True
        else:
            loss_new = loss

        return is_ID, batch_mean, batch_std, weighted_prompts, weights, loss, loss_new

    @torch.enable_grad()
    def forward(self, x):
        is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new = self._eval_coreset(x)
        
        if not is_ID and len(self.coreset) == self.max_num_prompts:
            is_ID = True
        
        if is_ID:
            for _ in range(self.E_ID):
                self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
                optimizer = torch.optim.AdamW([self.model.prompts], lr=self.cfg.OPTIM.LR)
                outputs, loss, batch_mean, batch_std = forward_and_adapt(x, self.model, optimizer, self.lamda, self.train_info)
            self._update_coreset(weights, batch_mean, batch_std)

        else:
            load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
            self.model.prompts.requires_grad_(True)
            self.optimizer = torch.optim.AdamW([self.model.prompts], lr=self.cfg.OPTIM.LR)
            
            for _ in range(self.E_OOD):
                outputs, loss, _, _ = forward_and_adapt(x, self.model, self.optimizer, self.lamda, self.train_info)

            self.coreset.append([batch_mean, batch_std, self.model.prompts.clone().detach().cpu()])
        
        return {'output':outputs, 'loss':loss, '#prototype': len(self.coreset)}
        
    
    def obtain_src_stat(self, data_path, num_samples=5000):

        # Caching the source features, don't need to re-compute for every runs
        stat_dir_path = os.path.join(self.cfg.CKPT_DIR, f"dpcore")
        os.makedirs(stat_dir_path, exist_ok=True)

        if self.cfg.CORRUPTION.DATASET == 'ccc':
            fname = os.path.join(stat_dir_path, "imagenet_c_train_info.pth")
        else:

            fname = os.path.join(stat_dir_path, f"{self.cfg.CORRUPTION.DATASET}_train_info.pth")

        if os.path.exists(fname):
            logger.info("Loading source stats...")
            self.train_info = torch.load(fname)

        else:

            num = 0
            features = []
            import timm
            from torchvision.datasets import ImageNet, STL10
            from tqdm import tqdm
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
            data_config = timm.data.resolve_model_data_config(net)
            src_transforms = timm.data.create_transform(**data_config, is_training=False)
            src_dataset = ImageNet(root=f"{data_path}/imagenet/images", split= 'train', transform=src_transforms)
            src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=64, shuffle=True)
            
            with torch.no_grad():
                for _, dl in tqdm(enumerate(src_loader), total=len(src_loader), desc="Processing Images"):
                    images = dl[0].cuda()
                    feature = self.model.forward_raw_features(images)
                    
                    output = self.model(images)
                    ent = softmax_entropy(output)
                    selected_indices = torch.where(ent < math.log(1000)/2-1)[0]
                    feature = feature[selected_indices]
                    
                    features.append(feature[:, 0])
                    num += feature.shape[0]
                    if num >= num_samples:
                        break

                features = torch.cat(features, dim=0)
                features = features[:num_samples, :]
                print(f'Source Statistics computed with {features.shape[0]} examples.')
                self.train_info = torch.std_mean(features, dim=0)

                torch.save(self.train_info, fname)

        
            del features
    

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def forward_and_get_loss(images, model:PromptViT, lamda, train_info, with_prompt=False):
    if with_prompt:
        cls_features = model.forward_features(images)[:, 0]
    else:
        cls_features = model.forward_raw_features(images)[:, 0]
    

    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    # std_mse, mean_mse = criterion_mse(batch_std, train_info[0].cuda()), criterion_mse(batch_mean, train_info[1].cuda())
    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    
    loss = lamda * std_loss + mean_loss
    
    # output = model.vit.forward_head(raw_features)

    return loss, batch_mean, batch_std

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def calculate_weights(coreset, batch_mean, batch_std, lamda, tau):
    mean_tensor = torch.stack([p[0] for p in coreset])
    std_tensor = torch.stack([p[1] for p in coreset])
    assert mean_tensor.shape[1] == 768 and mean_tensor.shape[0] == len(coreset)
    
    mean_match = torch.norm(batch_mean - mean_tensor, p=2, dim=1)
    std_match = torch.norm(batch_std - std_tensor, p=2, dim=1)
    
    match_loss = mean_match + lamda *  std_match
    weights = torch.nn.functional.softmax(-match_loss/tau, dim=0)
    # weights = weights.unsqueeze(-1).unsqueeze(-1)
    # print(f'weights: {weights}, sum: {weights.sum().item()}, loss: {match_loss}')
    # print(f'weights: {weights.tolist()}')
    return weights.detach().cpu()

def config_model(model, num_prompts):
    model = PromptViT(model, num_prompts)
    model.cuda()
    model.eval()
    model.requires_grad_(False)
    
    if num_prompts > 0:
        model.prompts.requires_grad_(True)

    return model

@torch.enable_grad()
def forward_and_adapt(x, model: PromptViT, optimizer, lamda, train_info):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    features = model.forward_features(x)
    cls_features = features[:, 0]
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    # std_mse, mean_mse = criterion_mse(batch_std, train_info[0].cuda()), criterion_mse(batch_mean, train_info[1].cuda())

    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    loss = lamda * std_loss + mean_loss
    
    # output = model.vit.head(cls_features)
    output = model.vit.forward_head(features)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output