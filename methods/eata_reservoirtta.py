"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""

import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import TTAMethod
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from methods.reservoirtta_utils import Plug_in_Bowl
from copy import deepcopy

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class EATA_ReservoirTTA(TTAMethod):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.num_samples_update_1 = 0  # number of samples after first filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after second filtering, exclude both unreliable and redundant samples
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = []  # the moving average of probability vector (Eqn. 4)
        for _ in range(cfg.RESERVOIRTTA.MAX_NUM_MODELS):
            self.current_model_probs.append(None)
        self.fisher_alpha = cfg.EATA.FISHER_ALPHA  # trade-off \beta for two losses (Eqn. 8)

        # setup loss function
        self.softmax_entropy = Entropy()

        if self.fisher_alpha > 0.0 and self.cfg.SOURCE.NUM_SAMPLES > 0:
            batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
            _, fisher_loader = get_source_loader(
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=self.model.model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                batch_size=batch_size_src,
                train_split=False,
                num_samples=cfg.SOURCE.NUM_SAMPLES,    # number of samples for ewc reg.
                percentage=cfg.SOURCE.PERCENTAGE,
                workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count())
            )
            self.update_fisher(fisher_loader)
        else:
            logger.info("Not using EWC regularization. EATA decays to ETA!")
            self.fishers = None


        ####################### Reservoir Start #######################

        self.reservoir = Plug_in_Bowl(cfg, self.img_size[0], self.params, 
                                 student_optimizer=self.optimizer,
                                 student_model=self.model
                                 )
        self.reservoir_output = {}

        ####################### Reservoir End #######################
        # note: reduce memory consumption by only saving normalization parameters
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.src_model, self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def update_fisher(self, loader: DataLoader):
        ewc_optimizer = torch.optim.SGD(self.params, 0.001)
        self.fishers = {} # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        train_loss_fn = nn.CrossEntropyLoss().to(self.device)
        for iter_, batch in enumerate(loader, start=1):
            images = batch[0].to(self.device, non_blocking=True)
            outputs = self.model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(loader):
                        fisher = fisher / iter_
                    self.fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("Finished computing the fisher matrices...")
        del ewc_optimizer

    def loss_calculation(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]

        ####################### Reservoir Start #######################
        self.reservoir_output = self.reservoir.clustering(imgs_test)
        
        with torch.no_grad():
            self.reservoir(ensembling=True, which_model='student')  # Sets student to be current shift model
            ensembled_outputs = self.model(imgs_test)               # TODO: change this so we take from self.reservoir_output

        self.reservoir(ensembling=False, which_model='student')

        self.optimizer.load_state_dict(self.reservoir.student.optimizer_reservoir[self.reservoir.model_idx])
        self.optimizer.zero_grad()
        ####################### Reservoir End #######################

        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs)

        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        entropys = entropys[filter_ids_1]

        # filter redundant samples
        if self.current_model_probs[self.reservoir.model_idx] is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs[self.reservoir.model_idx].unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs[self.reservoir.model_idx], outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs[self.reservoir.model_idx], outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = self.softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss

        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.current_model_probs[self.reservoir.model_idx] = updated_probs
        perform_update = len(entropys) != 0

        return ensembled_outputs, loss, perform_update

    @torch.enable_grad()
    def forward_and_adapt(self, x, is_source: bool = False):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, perform_update = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if perform_update and not is_source:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, perform_update = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if perform_update and not is_source:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            if perform_update and not is_source:
                ####################### Reservoir Start #######################
                self.reservoir.update_kth_model(self.optimizer, which_model='student', which_part='params')
                ####################### Reservoir End #######################

        return outputs


    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.current_model_probs = []  # the moving average of probability vector (Eqn. 4)
        for _ in range(self.cfg.RESERVOIRTTA.MAX_NUM_MODELS):
            self.current_model_probs = None

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with eata."""
        # train mode, because eata optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what eata updates
        self.model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
