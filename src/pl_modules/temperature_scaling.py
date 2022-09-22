# Author: Hendrik Mehrtens
# Date: 18.01.2021
# License: CC BY-SA: Creative Commons Attribution-ShareAlike


import torch
from torch import nn, optim
from torch.nn import functional as F

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from tqdm import tqdm

from einops import repeat, rearrange, parse_shape
from ..utils.einop import einop

""" TemperatureScalingCallback Class

	A PyTorch-Lightning callback that implements temperature scaling into your model in a classification task. 
	Automatically computes the temperature on_fit_end and optionally on_test_start and on_predict_start
	and applies it during testing and prediction, by decorating a 'forward' function (can be named differently), that needs to return non-softmaxed outputs. 
	
	This parametrization allow to also use a model, whichs forward method already outputs a softmax, by implemting an intermediate function into the model. 
		
	Supports automatic checkpointing and loading. 
	
	Details: The Temperature is automatically computed on_fit_end. If a validation_loader is provided it is also computed on_test_start and on_predict_start. Set compute_once to only compute a temperature
		if it has not been computed so far. Unless manual is set to True, temperature_scaling is automatically enables during testing and prediction.
	
	
	Args:
		 module (nn.Module -> includes pl_module): A nn.Module (includes a pytorch_lightning.LightningModule), where temperature scaling is applied to.
		 function_name (str): The name of the function, that returns the outputs. Defaults to 'forward'. Can however also be set to other values, even recusivly for sub-attributes. 
			 Example:
				LightningModuel: litModule
					nn.Module: net
						function: forward
						function: forward_without_softmax
			
			You can now create the Callback with these signatures:
			TemperatureScalingCallback(litModule, "net.forward_without_softmax", ...)
			TemperatureScalingCallback(net, "forward_without_softmax", ...)
			
			We will call .to(device) on the provided module parameter, so make sure the function_name function only depends on the parameters of the provided module.
		temperature (float): You can pre-provide a temperature-value that then will be used.
		compute_once (bool, Default:True): Only computes temperature if it is not set to far.
		
		validation_loader (torch.utils.data.Dataloader): If provided will use this dataloader for temperature computation. Otherwise will try to use the validation_dataloader from the Trainer. 
			Needed if no temperature is set and you do not call fit() before test() or predict().
		
		manual(bool, Default: False): Disables all automatic temperature computation and temperature_scaling activation and deactivation.
		
"""


@CALLBACK_REGISTRY
class TemperatureScalingCallback(Callback):
    def __init__(
        self,
        function_name="forward",
        number_temps=1,
        temperature=None,
        compute_once=True,
        validation_loader=None,
        manual=False,
    ):
        super(TemperatureScalingCallback, self).__init__()

        # self.wrapped_module = module

        self.func_name = function_name
        self.number_temp = number_temps

        if temperature is not None:
            assert len(temperature) == number_temps

        self.temperature = temperature
        self.enabled = False
        self.manual = manual
        self.compute_once = compute_once

        # Optional. If set we will recompute the temperature in each Trainer.test(...) call
        self.validation_loader = validation_loader

    # Core functions
    #################

    def update_temperature(self):

        if self.temperature is not None and self.compute_once:
            return

        # Determine how to get the val_dataloader
        if self.validation_loader is not None:
            val = self.validation_loader
        elif hasattr(self.trainer, "datamodule"):
            val = self.trainer.datamodule.val_dataloader()
        elif hasattr(self.trainer, "val_dataloaders"):
            if len(self.trainer.val_dataloaders) > 1:
                raise NotImplementedError
            val = self.trainer.val_dataloaders[0]
        else:
            raise NotImplementedError

        # TODO Find out how to get the device from the trainer ... it only uses accelarators.. mh
        # self.wrapped_module.device only returns cpu, which is strange. I guess it is only on the GPU during the loops, but the compute_temperature is called \
        # on_fit_end and on_test_start, on_predict_start... So far it is hardcoded and that should not be.
        # Ideally we would like to automate this with PyTorch Lightning. Also the loop in compute_temperature...
        device = self.wrapped_module.device
        self.temperature = compute_temperature(self.org_function, val, device, num_outputs=self.number_temp)

    def enable_temp_scaling(self):

        if self.temperature is None:
            raise Exception("Enabled temperature_scaling before computing or setting a temperature!")

        rsetattr(self.wrapped_module, self.func_name, self.temp_scaled_function)
        self.enabled = True

    def disable_temp_scaling(self):
        rsetattr(self.wrapped_module, self.func_name, self.org_function)
        self.enabled = False

    #################
    # -------------------------------------------------------------------------------------------------------------------

    # Helper functions
    #################

    def __temp_scale_forward_decorator(self, function):
        def wrapper(*args, **kwargs):
            out = function(*args, **kwargs)
            if isinstance(out, list):
                for i in range(len(self.temperature)):
                    out[i] /= self.temperature[i]
            else:
                out /= self.temperature

            return out

        return wrapper

    # Wrap the forward function of the lightning module
    def __hook_variables(self, trainer, module):
        self.trainer = trainer
        self.wrapped_module = module

        assert rhasattr(self.wrapped_module, self.func_name)

        # By swapping these two functions we can enable and disable temperature scaling
        self.org_function = rgetattr(self.wrapped_module, self.func_name)
        self.temp_scaled_function = self.__temp_scale_forward_decorator(rgetattr(self.wrapped_module, self.func_name))

    #################

    # -------------------------------------------------------------------------------------------------------------------

    # Compute temperature
    #################
    """
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.__hook_variables(trainer, pl_module)
        if not self.manual:
            self.update_temperature()
            # self.log("temperature", self.temperature) # Cannot log on_fit_end
            print("Computed temperature is %.4f" % self.temperature)
    """
    # If self.validation_loader is defined also compute on test_start
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.__hook_variables(trainer, pl_module)

        if self.validation_loader is not None and not self.manual:
            self.update_temperature()
            # self.log("temperature", self.temperature)
            print("Computed temperature is %.4f" % self.temperature.item())

    # If self.validation_loader is defined also compute on predict_start
    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.__hook_variables(trainer, pl_module)

        if self.validation_loader is not None and not self.manual:
            self.update_temperature()
            # Cannot log in predict phase
            print("Computed temperature is %.4f" % self.temperature.item())

    #################
    # -------------------------------------------------------------------------------------------------------------------

    # Wrap forward to apply temperature scaling
    #################

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.manual:
            self.enable_temp_scaling()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.manual:
            self.disable_temp_scaling()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.manual:
            self.enable_temp_scaling()

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs) -> None:
        if not self.manual:
            self.disable_temp_scaling()

    #################
    # -------------------------------------------------------------------------------------------------------------------

    # Removed this
    # Save stating
    #################
    """
    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.func_name = callback_state["func_name"]
        self.temperature = callback_state["temperature"]
        self.enabled = callback_state["enabled"]
        self.manual = callback_state["manual"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        save_dict = {}
        save_dict["func_name"] = self.func_name
        save_dict["temperature"] = self.temperature
        save_dict["enabled"] = self.enabled
        save_dict["manual"] = self.manual

        return save_dict
    """
    #################


def compute_temperature(model, valid_loader, device="cpu", num_outputs=1):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """

    print("Computing temperature")

    _has_init = False
    temperature = None
    logits_list = None
    labels_list = None
    nll_criterion = F.cross_entropy

    def _init_num_outputs():

        # First time I ever needed this...
        nonlocal _has_init, temperature, logits_list, labels_list

        temperature = torch.ones(num_outputs).to(device)
        temperature.requires_grad = True
        # First: collect all the logits and labels for the validation set
        logits_list = [[] for _ in range(num_outputs)]
        labels_list = []

    if num_outputs > 0:
        _init_num_outputs()

    with torch.no_grad():
        for input, label in tqdm(valid_loader):
            input = input.to(device)
            logits = model(input)

            if not isinstance(logits, list):
                logits = [logits]

            if num_outputs == -1:
                num_outputs = len(logits)
                _init_num_outputs()

            for i, logit in enumerate(logits):
                logits_list[i].append(logit.detach().clone())
                del logit

            labels_list.append(label.detach().clone())
            del logits
            del label

        for i in range(len(logits_list)):
            logits_list[i] = torch.cat(logits_list[i])

        logits = torch.stack(logits_list, dim=0).to(device)
        labels = torch.cat(labels_list).to(device)

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    # Size

    def eval():
        optimizer.zero_grad()

        # Mit Einops
        reshape_logits = rearrange(logits, "m b c -> (m b) c")
        reshape_labels = repeat(labels, "b -> (m b)", m=num_outputs)
        reshape_temperature = repeat(temperature, "m -> (m b) c", **parse_shape(logits, "_ b c"))

        loss = nll_criterion(reshape_logits / reshape_temperature, reshape_labels)
        loss.backward()
        return loss

        # Vergleich pures PyTorch
        # reshape_logits = logits.view(logits.shape[0] * logits.shape[1], -1)
        # reshape_labels = labels.repeat(num_outputs)
        # reshape_temperature = (
        #    temperature.repeat_interleave(logits.shape[1]).unsqueeze(1).expand(logits.size(0), logits.size(1))
        # )

        # loss = nll_criterion(reshape_logits / reshape_temperature, labels)
        # loss.backward()
        # return loss

        # loss = nll_criterion(
        #    logits / temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)),
        #    labels,
        # )
        # loss.backward()
        # return loss

    optimizer.step(eval)

    return temperature


def compute_temperature_from_list(preds, labels, num_outputs: int = 1):
    temperature = torch.ones(num_outputs)
    temperature.requires_grad = True
    nll_criterion = F.cross_entropy

    logits = torch.Tensor(preds)
    labels = torch.Tensor(labels)

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    # Size

    def eval():
        optimizer.zero_grad()

        # Mit Einops
        reshape_logits = rearrange(logits, "m b c -> (m b) c")
        reshape_labels = repeat(labels, "b -> (m b)", m=num_outputs)
        reshape_temperature = repeat(temperature, "m -> (m b) c", **parse_shape(logits, "_ b c"))

        loss = nll_criterion(reshape_logits / reshape_temperature, reshape_labels)
        loss.backward()
        return loss

        # Vergleich pures PyTorch
        # reshape_logits = logits.view(logits.shape[0] * logits.shape[1], -1)
        # reshape_labels = labels.repeat(num_outputs)
        # reshape_temperature = (
        #    temperature.repeat_interleave(logits.shape[1]).unsqueeze(1).expand(logits.size(0), logits.size(1))
        # )

        # loss = nll_criterion(reshape_logits / reshape_temperature, labels)
        # loss.backward()
        # return loss

        # loss = nll_criterion(
        #    logits / temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)),
        #    labels,
        # )
        # loss.backward()
        # return loss

    optimizer.step(eval)

    return temperature


###########
# Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
import functools


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr, *args):
    try:
        rgetattr(obj, attr, *args)
        return True
    except:
        return False


##############
# Alternativly use the magicattr package (pip install magicattr)


# Taken from the temp_scaling repository
'''
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
'''
