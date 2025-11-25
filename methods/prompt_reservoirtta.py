from utils.registry import ADAPTATION_REGISTRY
import torch
from .vpt import PromptViT
import os
import logging
import torch.nn as nn
from copy import deepcopy
import math
from methods.reservoirtta_utils import Plug_in_Bowl

logger = logging.getLogger(__name__)



@ADAPTATION_REGISTRY.register()
class Prompt_ReservoirTTA(nn.Module):
    def __init__(self, cfg, model, num_classes, scheduler: str = None):
        super().__init__(scheduler=scheduler)
        self.cfg = cfg

        self.lamda = 1.0
        self.temp_tau = cfg.DPCORE.TEMP_TAU
        self.ema_alpha = cfg.DPCORE.EMA_ALPHA
        self.thr_rho = cfg.DPCORE.THR_RHO
        self.E_ID = 1
        self.E_OOD = cfg.DPCORE.E_OOD
        
        self.model = config_model(model, cfg.DPCORE.NUM_PROMPTS)
        self.optimizer = torch.optim.AdamW([self.model.prompts], lr=cfg.OPTIM.LR)
        
        self.get_number_trainable_params([self.model.prompts])
        
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.coreset = []

        self.obtain_src_stat(cfg.DATA_DIR, cfg.DPCORE.SRC_NUM_SAMPLES)

        ####################### Reservoir Start #######################

        self.reservoir = Plug_in_Bowl(cfg, 224, [self.model.prompts], 
                                 student_optimizer=self.optimizer,
                                 student_model=self.model
                                 )
        self.reservoir_output = {}
        ####################### Reservoir End #######################


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

    @torch.enable_grad()
    def forward(self, x):
        
        ####################### Reservoir Start #######################
        self.reservoir_output = self.reservoir.clustering(x)

        with torch.no_grad():
            self.reservoir(ensembling=True, which_model='student')
            ensembled_outputs = self.model(x)

        self.reservoir(ensembling=False, which_model='student')

        self.optimizer.load_state_dict(self.reservoir.student.optimizer_reservoir[self.reservoir.model_idx])
        self.optimizer.zero_grad()
        ####################### Reservoir End #######################

        
        if self.reservoir_output['new_cluster']:
            # optimizer = torch.optim.AdamW([self.model.prompts], lr=0.1)
            from tqdm import tqdm

            progress_bar = tqdm(range(self.E_OOD), desc="Adapting Model")
            for _ in progress_bar:
                outputs, loss, _, _ = forward_and_adapt(x, self.model, self.optimizer, self.lamda, self.train_info)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")  # 显示 loss，保留 4 位小数


        else:
            #optimizer = torch.optim.AdamW([self.model.prompts], lr=0.1)
            outputs, loss, batch_mean, batch_std = forward_and_adapt(x, self.model, self.optimizer, self.lamda, self.train_info)


        ####################### Reservoir Start #######################
        self.reservoir.update_kth_model(self.optimizer, which_model='student', which_part='params')
        ####################### Reservoir End #######################
        
        return {'ensembled_outputs': ensembled_outputs, 'output': outputs, 'loss' : loss.item(),  **self.reservoir_output}
        
    
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