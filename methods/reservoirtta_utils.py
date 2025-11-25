import torch
import torch.nn as nn
import math

from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from copy import deepcopy
import torch.nn.functional as F

from datasets.data_loading import get_source_loader
from tqdm import tqdm

import math
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from collections import OrderedDict
import random
import logging

logger = logging.getLogger(__name__)
###################################################################################################################
###################################################################################################################
###################################################################################################################
###########################################    Plug in Reservoir       ############################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
class Plug_in_Bowl(torch.nn.Module):
    """Plug-in Reservoir for the ReservoirTTA method.

    This class implements a plug-in reservoir that can be used to store and manage model parameters
    for different domains in a reservoir-based transfer learning setup.
    """
    def __init__(self, cfg, img_size, 
                 ptr_student_params, 
                 student_optimizer=None,
                 student_model=None
                 ):
        """Initialize the Plug-in Reservoir.

        Args:
            cfg: Configuration object containing hyperparameters and settings.
            img_size: Size of the input images.
            ptr_student_params: Parameters of the student model to be used in the reservoir.
            student_optimizer: Optimizer for the student model.
            student_model: The student model itself.
        """
        super().__init__()

        self.cfg = cfg

        #################################################################################
        logger.info("#############################################")
        logger.info("###########  Hyper-parameters of ReservoirTTA ######################")
        self.emsembling = cfg.RESERVOIRTTA.ENSEMBLING

        if cfg.RESERVOIRTTA.SIZE_OF_BUFFER == 1:
            self.memory_updates = 'fifo'
        else:
            self.memory_updates = cfg.RESERVOIRTTA.SAMPLING
        self.progressive_update = cfg.RESERVOIRTTA.PROGRESSIVE_UPDATE
        self.accelerate_clustering = cfg.RESERVOIRTTA.SOURCE_BUFFER

        logger.info(f"ensembling = {self.emsembling}")
        logger.info(f"sampling strategy = {self.memory_updates}")
        logger.info(f"progressive_update = {self.progressive_update}")
        logger.info(f"initialize buffer with source features = {self.accelerate_clustering}")
        logger.info(f"use style idex = {cfg.RESERVOIRTTA.STYLE_IDX}")
        
        hyper_parameters = dict()
        hyper_parameters['max_num_of_reservoirs'] = cfg.RESERVOIRTTA.MAX_NUM_MODELS
        hyper_parameters['memory_bank_size'] = cfg.RESERVOIRTTA.SIZE_OF_BUFFER
        hyper_parameters['quantile_thr'] = cfg.RESERVOIRTTA.QUANTILE_THR
        hyper_parameters['init'] = cfg.RESERVOIRTTA.INIT
        
        hyper_parameters['reservoir_size_per_domain'] = int(cfg.RESERVOIRTTA.SIZE_OF_BUFFER / cfg.RESERVOIRTTA.MAX_NUM_MODELS)

        for key, value in hyper_parameters.items():
            logger.info(f"{key} = {value}")
        
        ##############################################################################
    
        self.num_reservoirs = 1
        self.max_num_reservoirs = hyper_parameters['max_num_of_reservoirs']
        self.quantile_thr = hyper_parameters['quantile_thr']
        self.init_method = hyper_parameters['init']

        # Other Attributes
        self.model_idx = 0
        self.model_prob = None
        self.model_idx_prob = 0.0
        self.cluster_losses = {}

        assert self.memory_updates in ["fifo", "replace", "reservoir"]
        
        # Define Style Extractor
        self.domain_extractor = StyleExtractor(img_size, style_idx=cfg.RESERVOIRTTA.STYLE_IDX, style_format=cfg.RESERVOIRTTA.STYLE_FORMAT).cuda()
        with torch.no_grad():
            dummy_img = torch.rand(1, 3, img_size, img_size).cuda()
            dummy_style = self.domain_extractor(dummy_img)
            style_shape = dummy_style.size(1) # C

        # Define Online Clustering
        source_style, thr = self.initialize()
        self.style_shape = style_shape

        self.online_cluster = MI_Uniform(self.num_reservoirs, style_shape, thr, hyper_parameters, source_style,
                                            self.memory_updates, self.max_num_reservoirs).cuda()
        if self.accelerate_clustering:
            self.initialize_queue_uniform()
        else:
            logger.info("Initialize with empty queue")


        # Make Reservoir for Student
        self.student = Plug_in_Reservoir(self.num_reservoirs, ptr_student_params, student_optimizer, student_model, self.emsembling, self.init_method)

        logger.info("#############################################")

    @torch.no_grad()
    def initialize(self,):
        mean_style = None
        all_feats = torch.Tensor().cuda()
        # initialize the buffer using source dataset
        _, source_loader = get_source_loader(
            dataset_name=self.cfg.CORRUPTION.DATASET,
            preprocess=None,
            data_root_dir=self.cfg.DATA_DIR,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            train_split= self.cfg.CORRUPTION.DATASET not in ["imagenet_c", "ccc"],
            num_samples=self.cfg.SOURCE.NUM_SAMPLES
        )
        while len(all_feats) < 1000:
            for data in source_loader:
                x = data[0].cuda()
                domain_vec = self.domain_extractor(x)
                all_feats = torch.cat([all_feats, domain_vec])
                if len(all_feats) >= 1000:
                    break
            logger.info(f"Computed {len(all_feats)}/{1000} features")
                
        mean_style = all_feats.mean(0, keepdim=True)
        pairwise_dists = torch.cdist(all_feats, all_feats)
        upper_tri_dists = pairwise_dists.triu(diagonal=1)
        valid_dists = upper_tri_dists[upper_tri_dists > 0]
        # thr = pairwise_dists.mean() + self.cfg.RESERVOIRTTA.STD_LAMBDA*pairwise_dists.std()
        thr = torch.quantile(valid_dists, q=self.quantile_thr)

        return mean_style.cpu(), thr
    
    @torch.no_grad()
    def initialize_queue_uniform(self):
        # initialize the buffer using source dataset
        _, source_loader = get_source_loader(
            dataset_name=self.cfg.CORRUPTION.DATASET,
            preprocess=None,
            data_root_dir=self.cfg.DATA_DIR,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            train_split= self.cfg.CORRUPTION.DATASET not in ["imagenet_c", "ccc"],
            num_samples=self.cfg.SOURCE.NUM_SAMPLES
        )
        pbar = tqdm(source_loader, dynamic_ncols=True)   
        
        for data in pbar:
            with torch.no_grad():
                x = data[0].cuda()
                domain_vec = self.domain_extractor(x)
                self.online_cluster.update_reservoir(domain_vec)
        logger.info(f"The queue is initialized with {self.online_cluster.reservoir_feats.shape[0]} soure features")

    def clustering(self, x):
            
        with torch.no_grad():
            style_vector = self.domain_extractor(x)
            
        model_idx, model_prob, cluster_losses, new_cluster = self.online_cluster.update(style_vector, len(x)==self.cfg.TEST.BATCH_SIZE)
        if new_cluster:
            self.num_reservoirs += 1
            logger.info(f"New cluster detected -- K={self.num_reservoirs}")
            self.student.add_reservoir(x)

        self.model_idx = model_idx
        self.model_prob = model_prob
        self.model_idx_prob = model_prob.amax()
        self.cluster_losses = cluster_losses

        return {'model_idx':self.model_idx, 'model_idx_prob':self.model_idx_prob, 'new_cluster': new_cluster, **cluster_losses}
    
    @torch.no_grad()
    def forward(self, ensembling=False, which_model='student', model_idx=None, model_prob=None):
        assert which_model in ['student', 'ema', 'anchor']

        if model_idx is None:
            model_idx = self.model_idx
            model_prob = self.model_prob

        if which_model == 'student':
            self.student(model_idx, model_prob, ensembling)
        else:
            raise ValueError("incorrect input")

    def update_kth_model(self, optimizer=None, which_model='student', which_part='params', model_idx=None, model_prob=None):
        assert which_model in ['student', 'ema', 'anchor'], print(which_model)
        assert which_part in ['params', 'stats', 'all']

        if model_idx is None:
            model_idx = self.model_idx

        if self.progressive_update: 
            if model_prob is None:
                update_speed = self.model_idx_prob
            else:
                update_speed = model_prob
        else:
            update_speed = 1.0

        self.student.update_kth_model(model_idx, update_speed, optimizer, which_part)


    def reset_kth_model(self, optimizer_state=None, which_model='student', which_part='params', model_idx=None,):
        assert which_model in ['student', 'ema', 'anchor'], print(which_model)
        assert which_part in ['params', 'stats', 'all']

        if model_idx is None:
            model_idx = self.model_idx

        self.student.reset_kth_model(model_idx, optimizer_state, which_part)



    def compute_model_distance(self):
        loss = 0
        n_items = 0

        for curr_p, source_p in zip(self.student.ptr_bn_params, self.student.bn_params):
            loss += F.mse_loss(curr_p, source_p.data)
            n_items += 1
        
        return loss / n_items
    
class Plug_in_Reservoir(torch.nn.Module):
    """Plug-in Reservoir for the ReservoirTTA method.

    This class implements a plug-in reservoir that can be used to store and manage model parameters
    for different domains in a reservoir-based transfer learning setup.
    """
    def __init__(self, num_reservoirs, ptr_bn_params, optimizer=None, model=None, ensembling=True, init_method="mi"):
        """Initialize the Plug-in Reservoir.

        Args:
            num_reservoirs: Number of reservoirs to maintain.
            ptr_bn_params: Pointer to the batch normalization parameters of the model.
            optimizer: Optimizer for the model.
            model: The model itself.
            ensembling: Whether to use ensembling of parameters.
            init_method: Method to initialize the reservoir ('source' or 'mi').
        """
        super().__init__()

        self.ensembling=ensembling
        self.model=model
        self.init_method=init_method
        self.softmax_entropy = Entropy()

        # Make reservoir for BN Params
        self.ptr_bn_params = ptr_bn_params 
        if self.ptr_bn_params is not None:
            with torch.no_grad():
                self.bn_params = clone_params(self.ptr_bn_params) 
            self.bn_params_reservoir = [ deepcopy(self.bn_params) for _ in range(num_reservoirs) ]

        # Optimizer
        self.optimizer_state = deepcopy(optimizer.state_dict())
        if optimizer is not None:
            self.optimizer_reservoir = [deepcopy(optimizer.state_dict()) for _ in range(num_reservoirs)]

    @torch.no_grad()
    def add_reservoir(self, x):
        if self.init_method == "source":
            self.bn_params_reservoir.append(clone_params(self.bn_params))
            self.optimizer_reservoir.append(deepcopy(self.optimizer_state))
        elif self.init_method == "mi":
            self.add_reservoir_mi(x)
        else:
            raise NotImplementedError(f"Init Method {self.init_method} not implemented!")


    @torch.no_grad()
    def add_reservoir_mi(self, x):
        best_idx = 0
        min_ent = torch.inf
        for i in range(len(self.bn_params_reservoir)):
            self.forward(i, None, emsembling=False)
            pred = self.model(x)
            ent = self.softmax_entropy(pred).mean()
            probs = F.softmax(pred, dim=-1)
            avg_probs = probs.mean(dim=0)
            class_marginal = torch.sum(avg_probs * torch.log(avg_probs + 1e-8)) #  torch.mean((avg_probs - (1.0 / self.k)).abs())
            if ent+class_marginal < min_ent:
                best_idx = i
                min_ent = (ent+class_marginal).clone()

        logger.info(f"Initializing Model {len(self.bn_params_reservoir)} with Parameters of Model {best_idx}")
        self.bn_params_reservoir.append(clone_params(self.bn_params_reservoir[best_idx]))
        self.optimizer_reservoir.append(deepcopy(self.optimizer_reservoir[best_idx]))

    @torch.no_grad()
    def forward(self, model_idx, model_prob, emsembling=False):
        """Forward pass through the reservoir.

        Args:
            model_idx: Index of the model to forward through.
            model_prob: Probability of the model.
            emsembling: Whether to use ensembling of parameters.
        """
        assert model_idx < len(self.bn_params_reservoir), print(f"Model index {model_idx} out of range {len(self.bn_params_reservoir)}")
        if not emsembling or not self.ensembling:
            if self.ptr_bn_params is not None:
                set_bn_params(self.ptr_bn_params, self.bn_params_reservoir[model_idx])
        else:
            # ensembling params
            if self.ptr_bn_params is not None:
                ensembled_weights = []
                for curr_param_weights in zip(*self.bn_params_reservoir):
                    curr_param_weights = model_prob * torch.stack(curr_param_weights, dim=-1) # size of reservoirs
                    curr_param_ensemble = torch.sum(curr_param_weights, dim=-1)
                    ensembled_weights.append(curr_param_ensemble)
                set_bn_params(self.ptr_bn_params, ensembled_weights)


    def update_kth_model(self, model_idx, model_idx_prob, curr_optimizer=None, which_part='params'):
        """Update the parameters of the kth model in the reservoir.

        Args:
            model_idx: Index of the model to update.
            model_idx_prob: Probability of the model.
            curr_optimizer: Current optimizer state to update.
            which_part: Part of the model to update ('params', or 'all').
        """
        assert which_part in ['params', 'all'], print(which_part)
        k = model_idx
        p = model_idx_prob

        if which_part in ['params', 'all']:
            curr_params = self.ptr_bn_params
            for param_k_model, curr_param in zip(self.bn_params_reservoir[k], curr_params):
                param_k_model.mul_(1-p).add_((p) * curr_param)

        if curr_optimizer is not None:
            self.optimizer_reservoir[k] = curr_optimizer.state_dict()


    def reset_kth_model(self, model_idx, curr_optimizer_state=None, which_part='params'):
        """Reset the parameters of the kth model in the reservoir.

        Args:
            model_idx: Index of the model to reset.
            curr_optimizer_state: Current optimizer state to reset.
            which_part: Part of the model to reset ('params', or 'all').
        """
        assert which_part in ['params', 'all'], print(which_part)
        k = model_idx

        if which_part in ['params', 'all']:
            for param_k_model, curr_param in zip(self.bn_params_reservoir[k], self.bn_params):
                param_k_model.mul_(0.0).add_((1.0) * curr_param)

        if curr_optimizer_state is not None:
            self.optimizer_reservoir[k] = curr_optimizer_state


###################################################################################################################
###################################################################################################################
###################################################################################################################
###########################################    Helper Functions   #################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

def clone_params(input_params):
    params = []
    for p in input_params:
        params.append(p.data.clone())
    
    return params

def set_bn_params(bn_params, target_params):
    for curr_p, tgt_p in zip(bn_params, target_params):
        curr_p.data.zero_().add_(tgt_p)

def get_nested_attribute(model, attr_name):
    """Traverse the model to get a nested attribute."""
    parts = attr_name.split('.')
    current = model
    for part in parts:
        # If part is an integer (e.g., for list or Sequential indexing), convert it
        if part.isdigit():
            part = int(part)
            current = current[part]  # Access Sequential or list element
        else:
            current = getattr(current, part)  # Access attribute
    return current


def collect_bn_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

###################################################################################################################
###################################################################################################################
###################################################################################################################
###########################################     Style model    ####################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

def log_var(features):

    var = torch.var(features, dim=(0, 2, 3))
    var = torch.log(var)
    feat = var[None, ...]

    return feat

def mean(features):
    mean = features.mean(dim=[2, 3])
    return mean.mean(dim=0)[None, ...]

def var(features):
    var = torch.var(features, dim=(0, 2, 3))
    return var[None, ...]

def mean_var(features):
    # Compute channel-wise mean and log variance
    mean = features.mean(dim=[2, 3])  # [B, C]
    variance = features.var(dim=[2, 3]) + 1e-5  # [B, C], add epsilon for stability
    log_variance = torch.log(variance)
    
    # Concatenate mean and log variance to form the style vector
    style_vector = torch.cat([mean, log_variance], dim=1)  # [B, 2C]
    return style_vector.mean(dim=0)[None, ...]

def gram_matrix_diagonal(features):
    """
    Compute the diagonal of the Gram matrix.

    Args:
    - features: Tensor of shape [B, C, H, W] (Batch, Channels, Height, Width)

    Returns:
    - diagonal: Tensor of shape [B, C], the diagonal of the Gram matrix
    """
    B, C, H, W = features.shape
    features = features.view(B, C, -1)  # Flatten spatial dimensions: [B, C, H*W]
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram matrix: [B, C, C]
    diagonal = gram.diagonal(dim1=-2, dim2=-1)  # Extract diagonal elements: [B, C]
    return diagonal.mean(dim=0)[None, ...]


class StyleExtractor(torch.nn.Module):
    """Style Extractor for the ReservoirTTA method.

    This class extracts style features from images using a pre-trained VGG19 model.
    The style features can be computed in different formats such as LOGVAR, MEAN, VAR, MEAN_VAR, or GRAM.
    """
    def __init__(self, img_size=128, style_idx=[2, 5, 7], style_format="LOGVAR") -> None:
        """Initialize the Style Extractor.

        Args:
            img_size: Size of the input images (default: 128).
            style_idx: List of indices of VGG19 layers to extract style features from (default: [2, 5, 7]).
            style_format: Format of the style features to be extracted (default: "LOGVAR").
        """
        super().__init__()

        self.style_format = style_format

        style_functions = {
            'LOGVAR': log_var,
            'MEAN': mean,
            'VAR': var,
            'MEAN_VAR': mean_var,
            'GRAM': gram_matrix_diagonal
        }

        # Use dictionary to assign the function
        self.produce_style = style_functions.get(style_format)

        # If the provided style_format is invalid, raise an error
        if self.produce_style is None:
            raise ValueError(f"Invalid style_format: {style_format}. Must be one of {list(style_functions.keys())}.")
        else:
            logger.info(f"Use {self.style_format} to produce style features")

        max_style_idx = max(style_idx)
        # get vgg19 pre-trained to extract the domain of the images
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:max_style_idx+1].eval()
        vgg.requires_grad_(False)

        # replace inplace relu with normal relu
        modified_vgg = torch.nn.Sequential()

        for i, layer in enumerate(vgg.children()):
            if isinstance(layer, torch.nn.ReLU):
                layer = torch.nn.ReLU()
            
            modified_vgg.add_module(str(i), layer)
        
        self.vgg = modified_vgg

        self.img_size = img_size
        self.style_idx = style_idx

        # image size, transforms
        normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        normalization_std = torch.tensor([0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.Normalize(normalization_mean, normalization_std),
            ]
        )

        # register hooks to get features at different layers
        self.features = {}
        for s_idx in self.style_idx:
            self.vgg[s_idx].register_forward_hook(self.get_hook(s_idx))


    def compute_style_features(self):
        style_features = []
        for idx in self.style_idx:
            feat = self.produce_style(self.features[idx])
            style_features.append(feat)

        style_features = torch.cat(style_features, dim=1)
        return style_features


    @torch.no_grad()
    def forward(self, x):
        x = self.transform(x)
        self.vgg(x)
        return self.compute_style_features()

    
    def get_hook(self, layer_name):
        def hook(module, args, output):
            with torch.no_grad():
                self.features[layer_name] = output
        return hook



###################################################################################################################
###################################################################################################################
###################################################################################################################
###########################################     Online Clustering    ##############################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
class MI_Uniform(torch.nn.Module):
    """MI Uniform Reservoir for the ReservoirTTA method.

    This class implements a reservoir that uses mutual information to manage model parameters for different domains.
    It supports various memory update strategies and can dynamically create new centroids based on the input features.
    """
    def __init__(self, k, d, thr, hyper_parameters: dict, init_style=None, memory_updates='reservoir',
                    max_num_of_reservoirs=16) -> None:
        """Initialize the MI Uniform Reservoir.

        Args:
            k: Number of clusters (domains).
            d: Dimension of the feature space.
            thr: Threshold for creating new centroids.
            hyper_parameters: Dictionary containing hyperparameters for the reservoir.
            init_style: Initial style vector (optional).
            memory_updates: Strategy for updating the memory ('reservoir', 'fifo', or 'replace').
            max_num_of_reservoirs: Maximum number of reservoirs allowed.
        """
        super(MI_Uniform, self).__init__()
        self.k = k
        self.d = d
        self.thr = thr
        self.max_num_of_reservoirs = max_num_of_reservoirs
        self.memory_updates = memory_updates

        # you can change it to whatever you want
        self.reservoir_size_per_domain = hyper_parameters['reservoir_size_per_domain']
        self.count_reset_ratio = 10

        # used for tracking number of items seen for a domain
        self.total_counts = 0
        self.register_buffer(f"reservoir_feats", torch.empty(0, d))
        self.register_buffer("init_style", init_style if init_style is not None else torch.zeros(1, self.d))

        self.delta_centroids = torch.nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.optimizer = torch.optim.AdamW(params=[self.delta_centroids], lr=1e-4)


    @torch.no_grad()
    def update_reservoir(self, feats):
        """Update the reservoir with new features.

        Args:
            feats: Tensor of shape [B, d] containing the new features to be added to the reservoir.
        """
        if self.memory_updates == 'reservoir':
            centroids = self.delta_centroids + self.init_style
            reservoir = torch.cat([self.reservoir_feats, feats], dim=0)
            scores = -torch.cdist(reservoir, centroids)

            probs  = F.softmax(scores, dim=-1)
            with torch.no_grad():
                probs_models = probs[-1:]
                model_idx = torch.argmax(probs_models, dim=-1)[0]

                if self.reservoir_feats.size(0) < self.reservoir_size_per_domain * self.k:
                    self.reservoir_feats = torch.cat([self.reservoir_feats, feats], dim=0)
                else:
                    replace_index = random.randint(0, self.total_counts - 1)
                    if replace_index < self.reservoir_size_per_domain * self.k:
                        self.reservoir_feats[replace_index] = feats[0]

                self.total_counts += 1

                if self.total_counts > self.count_reset_ratio * self.reservoir_size_per_domain * self.k:
                    self.total_counts = self.reservoir_size_per_domain * self.k
        elif self.memory_updates == 'fifo':
            if self.reservoir_feats.size(0) < self.reservoir_size_per_domain * self.k:
                self.reservoir_feats = torch.cat([self.reservoir_feats, feats], dim=0)
            else:
                self.reservoir_feats = self.reservoir_feats[1:]
                self.reservoir_feats = torch.cat([self.reservoir_feats, feats], dim=0)
        elif self.memory_updates == 'replace':
            centroids = self.delta_centroids + self.init_style
            reservoir = torch.cat([self.reservoir_feats, feats], dim=0)
            scores = -torch.cdist(reservoir, centroids)

            probs  = F.softmax(scores, dim=-1)
            with torch.no_grad():
                probs_models = probs[-1:]
                model_idx = torch.argmax(probs_models, dim=-1)[0]
                if self.reservoir_feats.size(0) < self.reservoir_size_per_domain * self.k:
                    self.reservoir_feats = torch.cat([self.reservoir_feats, feats], dim=0)
                else:
                    probs_curr_idx  = probs[:-1, model_idx]

                    _, top_idxs = torch.topk(probs_curr_idx, k = (self.reservoir_feats.size(0) // self.k), sorted=True)
                    # repalce the farthest element
                    selected_idx = top_idxs[-1]
                    if probs_curr_idx[selected_idx] <= probs[-1, model_idx]:
                        self.reservoir_feats[selected_idx] = feats[0]


    def create_new_centroid(self, feats):
        self.delta_centroids = torch.nn.Parameter(torch.cat([self.delta_centroids.data, feats-self.init_style]), requires_grad=True)
        self.optimizer = torch.optim.AdamW(params=[self.delta_centroids], lr=1e-4)


    def return_centroid(self):
        return self.delta_centroids.detach().clone() + self.init_style
    
    def update(self, feats, is_valid_batch : bool = True):
        """Update the reservoir with new features and compute the loss.

        Args:
            feats: Tensor of shape [B, d] containing the new features to be added to the reservoir.
            is_valid_batch: Boolean indicating whether the batch is valid for updating the reservoir.
        
        Returns:
            model_idx: Index of the model that best matches the input features.
            probs_models: Probabilities of each model given the input features.
            cluster_losses: Dictionary containing entropy and class marginal losses.
            new_cluster: Boolean indicating whether a new cluster was created.
        """
        self.optimizer.zero_grad()
        # first compute the model component
        centroids = self.delta_centroids + self.init_style
        reservoir = torch.cat([self.reservoir_feats, feats], dim=0)
        scores = -torch.cdist(reservoir, centroids)

        # Create a new centroid if needed
        new_cluster = False
        if is_valid_batch:
            with torch.no_grad():
                min_dist = (-scores[-1]).amin()
                if min_dist > self.thr and ((self.k < self.max_num_of_reservoirs)):
                    self.k += 1
                    new_cluster = True
                    probs  = F.softmax(scores, dim=-1)
                    probs_models = probs[-1:]

            if new_cluster:
                self.create_new_centroid(feats)
                # Redo prediction
                centroids = self.delta_centroids + self.init_style
                scores = -torch.cdist(reservoir, centroids)

        # Update centroids
        probs  = F.softmax(scores, dim=-1)
        avg_probs = probs.mean(dim=0)
        log_probs = F.log_softmax(scores, dim=-1)
        entropy = torch.sum(-probs * log_probs, dim=-1).mean() 
        class_marginal = torch.sum(avg_probs * torch.log(avg_probs + 1e-8)) #  torch.mean((avg_probs - (1.0 / self.k)).abs()) 
        loss = entropy + class_marginal 

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            probs_models = probs[-1:]
            model_idx = torch.argmax(probs_models, dim=-1)[0]

            if self.reservoir_feats.size(0) < self.reservoir_size_per_domain * self.k:
                self.reservoir_feats = torch.cat([self.reservoir_feats, feats], dim=0)
            else:
                if self.memory_updates == 'reservoir':           
                    replace_index = random.randint(0, self.total_counts - 1)
                    if replace_index < self.reservoir_size_per_domain * self.k:
                        self.reservoir_feats[replace_index] = feats[0]

                    

                    if self.total_counts > self.count_reset_ratio * self.reservoir_size_per_domain * self.k:
                        self.total_counts = self.reservoir_size_per_domain * self.k
                
                elif self.memory_updates == 'fifo':    
                    self.reservoir_feats = self.reservoir_feats[1:]
                    self.reservoir_feats = torch.cat([self.reservoir_feats, feats], dim=0)

                elif self.memory_updates == 'replace':
 
                    probs_curr_idx  = probs[:-1, model_idx]

                    _, top_idxs = torch.topk(probs_curr_idx, k = (self.reservoir_feats.size(0) // self.k), sorted=True)
                    # repalce the farthest element
                    selected_idx = top_idxs[-1]
                    if probs_curr_idx[selected_idx] <= probs[-1, model_idx]:
                        self.reservoir_feats[selected_idx] = feats[0]
            self.total_counts += 1
            
        return model_idx, probs_models.detach(), {"entropy" : entropy.detach(), "cm" : class_marginal.detach(),}, new_cluster


    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # Get the standard state_dict
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Add the custom tensor
        state[prefix + 'reservoir_feats'] = self.reservoir_feats
        state[prefix + 'delta_centroids'] = self.delta_centroids
        state[prefix + 'init_style'] = self.init_style
        return state

    def load_state_dict(self, *args, state_dict, strict=True):
        # Load the custom tensor
        state_dict = dict(state_dict)
        self.reservoir = state_dict.pop('reservoir_feats')
        self.delta_centroids = state_dict.pop('delta_centroids')
        self.init_style = state_dict.pop('init_style')
        state_dict = OrderedDict(state_dict)
        super().load_state_dict(state_dict, strict)
