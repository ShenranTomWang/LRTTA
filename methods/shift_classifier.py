import torch
import torchvision.transforms as transforms
import logging, random
import torch.nn.functional as F
from collections import OrderedDict
logger = logging.getLogger(__name__)
from torchvision.models import vgg19, VGG19_Weights
from datasets.data_loading import get_source_loader
from .utils import log_var, mean, var, mean_var, gram_matrix_diagonal

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


class ShiftClassifier(torch.nn.Module):
    def __init__(self, cfg, img_size):
        super().__init__()
        hyper_parameters = dict()
        hyper_parameters['max_num_of_reservoirs'] = cfg.RESERVOIRTTA.MAX_NUM_MODELS
        hyper_parameters['memory_bank_size'] = cfg.RESERVOIRTTA.SIZE_OF_BUFFER
        hyper_parameters['quantile_thr'] = cfg.RESERVOIRTTA.QUANTILE_THR
        hyper_parameters['init'] = cfg.RESERVOIRTTA.INIT
        
        hyper_parameters['reservoir_size_per_domain'] = int(cfg.RESERVOIRTTA.SIZE_OF_BUFFER / cfg.RESERVOIRTTA.MAX_NUM_MODELS)
        self.max_num_reservoirs = hyper_parameters['max_num_of_reservoirs']
        if cfg.RESERVOIRTTA.SIZE_OF_BUFFER == 1:
            self.memory_updates = 'fifo'
        else:
            self.memory_updates = cfg.RESERVOIRTTA.SAMPLING
        
        self.domain_extractor = StyleExtractor(
            img_size, style_idx=cfg.RESERVOIRTTA.STYLE_IDX, style_format=cfg.RESERVOIRTTA.STYLE_FORMAT
        ).cuda()
        with torch.no_grad():
            dummy_img = torch.rand(1, 3, img_size, img_size).cuda()
            dummy_style = self.domain_extractor(dummy_img)
            style_shape = dummy_style.size(1)
        source_style, thr = self.initialize()
        self.style_shape = style_shape

        self.online_cluster = MI_Uniform(
            self.num_reservoirs, style_shape, thr, hyper_parameters,
            source_style, self.memory_updates, self.max_num_reservoirs
        ).cuda()
        
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
        thr = torch.quantile(valid_dists, q=self.quantile_thr)

        return mean_style.cpu(), thr

    def forward(self, x):
        with torch.no_grad():
            style_vector = self.domain_extractor(x)
            
        model_idx, model_prob, cluster_losses, new_cluster = self.online_cluster.update(style_vector, len(x)==self.cfg.TEST.BATCH_SIZE)
        if new_cluster:
            self.num_reservoirs += 1
            logger.info(f"New cluster detected -- K={self.num_reservoirs}")

        self.model_idx = model_idx
        self.model_prob = model_prob
        self.model_idx_prob = model_prob.amax()
        self.cluster_losses = cluster_losses

        return {'model_idx':self.model_idx, 'model_idx_prob':self.model_idx_prob, **cluster_losses}