import os
import re
import types
from typing import Any
from typing import Optional
from typing import Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from deepspeed.runtime.engine import DeepSpeedEngine as _DeepSpeedEngine
from deepspeed.runtime.pipe.engine import PipelineEngine as _PipelineEngine
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.utils import logger, log_dist

from deepspeed.git_version_info import version, git_hash, git_branch

try:
    from apex import amp
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    pass

def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    matched = re.search('^(\d+)\.(\d+)\.(\d+)', version_str)
    return int(matched.group(1)), int(matched.group(2)), int(matched.group(3))


# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch


def _load_checkpoint(self,
                     load_dir,
                     tag,
                     load_module_strict=True,
                     load_optimizer_states=True,
                     load_lr_scheduler_states=True):

    load_path = self._get_ckpt_name(load_dir, tag)

    if not os.path.exists(load_path):
        logger.warn(
            'Client provided checkpoint load path: {} does not exist ... skip checkpoint load'
                .format(load_path))
        return None, None

    logger.info(f'rank: {self.global_rank} loading checkpoint: {load_path}')
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)

    if isinstance(self.module, PipelineModule):
        # Pipeline parallelism uses this to load its own checkpoint files.
        self._curr_ckpt_path = os.path.join(load_dir, tag)

    self.load_module_state_dict(state_dict=checkpoint['module'],
                                strict=load_module_strict)
    if self.optimizer is not None and not self.zero_optimization():
        if self.fp16_enabled():
            self.optimizer.load_state_dict(
                checkpoint['optimizer'],
                load_optimizer_states=load_optimizer_states)
        elif load_optimizer_states:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
    if self.preconditioner is not None and 'preconditioner' in checkpoint:
        self.preconditioner.load_state_dict(checkpoint['preconditioner'])

    if load_lr_scheduler_states and self.lr_scheduler is not None:
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    self.csr_tensor_module_names = checkpoint['csr_tensor_module_names']
    self.global_steps = checkpoint['global_steps']
    self.global_samples = checkpoint.get('global_samples',
                                         self.global_steps * self.train_batch_size())
    self.skipped_steps = checkpoint['skipped_steps']
    self.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
    self.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']
    deepspeed_states = [
        'module',
        'optimizer',
        'lr_scheduler',
        'csr_tensor_module_names',
        'skipped_steps',
        'global_steps',
        'dp_world_size',
        'mp_world_size'
    ]
    client_state = {
        key: value
        for key,
            value in checkpoint.items() if not key in deepspeed_states
    }

    return load_path, client_state


def _save_checkpoint(self, save_dir, tag, client_state={}):

    save_path = self._get_ckpt_name(save_dir, tag)
    # A hack to save the checkpointing directory. Pipeline parallelism overrides
    # module_state_dict() and uses this path to save the model. module_state_dict()
    # then instead just returns None.
    self._curr_ckpt_path = os.path.join(save_dir, tag)

    state = dict(
        module=self.module_state_dict(),
        optimizer=self.optimizer.state_dict()
        if self.optimizer and not self.zero_optimization() else None,
        preconditioner=self.preconditioner.state_dict()
        if self.preconditioner is not None else None,
        lr_scheduler=self.lr_scheduler.state_dict()
        if self.lr_scheduler is not None else None,
        csr_tensor_module_names=self.csr_tensor_module_names,
        skipped_steps=self.skipped_steps,
        global_steps=self.global_steps,
        global_samples=self.global_samples,
        dp_world_size=self.dp_world_size,
        mp_world_size=self.mp_world_size,
    )
    state.update(client_state)

    log_dist(message=f'Saving model checkpoint: {save_path}', ranks=[0])
    # logger.info('Saving model checkpoint: {}'.format(save_path))
    torch.save(state, save_path)
    self._curr_save_path = None

def _take_model_step(self, lr_kwargs):
    if self.gradient_clipping() > 0.0:
        self.timers('_step_clipping').start()
        if not self.fp16_enabled() and not self.amp_enabled():
            self.clip_fp32_gradients()
        elif self.amp_enabled():
            # AMP's recommended way of doing clipping
            # https://nvidia.github.io/apex/advanced.html#gradient-clipping
            master_params = amp.master_params(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters=master_params,
                                           max_norm=self.gradient_clipping())
        self.timers('_step_clipping').stop()

    # store gradients
    if self.store_gradients:
        if self.store_gradients_cpu:
            self.stored_gradients = list([p.grad.clone().cpu() for p in self.module.parameters()])
        else:
            self.stored_gradients = list([p.grad.clone() for p in self.module.parameters()])

    if self.preconditioner is not None:
        if (
            (
                hasattr(self.optimizer, 'overflow') 
                and not self.optimizer.overflow
                # skip first step because overflow flag not set
                and self.global_steps > 0
            )
            or not hasattr(self.optimizer, 'overflow')
        ):
            self.timers('_step_precondition').start()
            self.preconditioner.step()
            self.timers('_step_precondition').stop()
        else:
            # We are skipping KFAC step for some reason so assume the
            # last batch was bad and get rid of it
            self.preconditioner.reset_batch()

    self.timers('_step_step').start()
    if self.zero_optimization_stage() == 1 and self.wall_clock_breakdown():
        self.optimizer.step(comms_timer=self.timers('comms'))
    else:
        self.optimizer.step()
    self.timers('_step_step').stop()

    self.timers('_step_zero_grad').start()
    # zero grad in basic optimizer could be unreliable and may not exhibit
    # the behaviour that we want
    if not self.zero_optimization() and not self.fp16_enabled(
    ) and not self.amp_enabled():
        self.zero_grad()
    else:
        self.optimizer.zero_grad()
    self.timers('_step_zero_grad').stop()

    report_progress = self.global_rank == 0 if self.global_rank else True

    self.timers('_step_check_overflow').start()
    # Check overlow here since in DS fp16 optimizer, the overflow is updated in above step() function.
    overflow = False
    if hasattr(self.optimizer, 'overflow'):
        overflow = self.optimizer.overflow

    if overflow:
        self.skipped_steps += 1
    else:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(**(lr_kwargs or {}))

    if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
        self._report_progress(self.global_steps + 1)
    self.timers('_step_check_overflow').stop()

    self.global_steps += 1
    self.global_samples += self.train_batch_size()


class DeepSpeedEngine(_DeepSpeedEngine):
    def __init__(self, **kwargs):
        self.preconditioner = kwargs.pop('preconditioner', None)
        super().__init__(**kwargs)

        # Monkeypatch loss scale into the preconditioner
        if hasattr(self.optimizer, 'loss_scale'):
            # Newer DeepSpeed uses loss_scale property
            grad_scaler = lambda: self.optimizer.loss_scale
        elif hasattr(self.optimizer, 'cur_scale'):
            # DeeperSpeed uses cur_scale attribute
            grad_scaler = lambda: self.optimizer.cur_scale
        else:
            assert False
            grad_scaler = None
        if grad_scaler is not None and self.preconditioner is not None:
            self.preconditioner.grad_scaler = grad_scaler
            for layer in self.preconditioner.layers.values():
                layer.grad_scaler = grad_scaler

        self._load_checkpoint = types.MethodType(_load_checkpoint, self)
        self._save_checkpoint = types.MethodType(_save_checkpoint, self)
        self._take_model_step = types.MethodType(_take_model_step, self)


class PipelineEngine(_PipelineEngine):
    def __init__(self, **kwargs):
        self.preconditioner = kwargs.pop('preconditioner', None)
        super().__init__(**kwargs)
        
        # Monkeypatch loss scale into the preconditioner
        if hasattr(self.optimizer, 'loss_scale'):
            # Newer DeepSpeed uses loss_scale property
            grad_scaler = lambda: self.optimizer.loss_scale
        elif hasattr(self.optimizer, 'cur_scale'):
            # DeeperSpeed uses cur_scale attribute
            grad_scaler = lambda: self.optimizer.cur_scale
        else:
            assert False
            grad_scaler = None
        if grad_scaler is not None and self.preconditioner is not None:
            self.preconditioner.grad_scaler = grad_scaler
            for _, layer in self.preconditioner._layers.values():
                layer.grad_scaler = grad_scaler

        self._load_checkpoint = types.MethodType(_load_checkpoint, self)
        self._save_checkpoint = types.MethodType(_save_checkpoint, self)
        self._take_model_step = types.MethodType(_take_model_step, self)

def initialize(args=None,
               model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer, Any]] = None,
               preconditioner: Optional[Any] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler, Any]] = None,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               config_params=None):
    """Initialize the DeepSpeed Engine.
    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.
        model: Required: nn.module class before apply any wrappers
        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.
        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.
        training_data: Optional: Dataset of type torch.utils.data.Dataset
        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods
        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()
        dist_init_required: Optional: None will auto-initialize torch.distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.
        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        config_params: Optional: Same as `config`, kept for backwards compatibility.
    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``
        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.
        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.
        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.
        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
             ranks=[0])
    assert model is not None, "deepspeed.initialize requires a model"

    if not isinstance(model, PipelineModule):
        engine = DeepSpeedEngine(args=args,
                                 model=model,
                                 optimizer=optimizer,
                                 preconditioner=preconditioner,
                                 model_parameters=model_parameters,
                                 training_data=training_data,
                                 lr_scheduler=lr_scheduler,
                                 mpu=mpu,
                                 dist_init_required=dist_init_required,
                                 collate_fn=collate_fn,
                                 config_params=config_params)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                preconditioner=preconditioner,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=model.mpu(),
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config_params=config_params)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)
