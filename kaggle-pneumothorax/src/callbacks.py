import logging
import os
from queue import PriorityQueue

import torch
from tensorboardX import SummaryWriter


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.runner = None
        self.metrics = None

    def set_runner(self, runner):
        self.runner = runner
        self.metrics = runner.metrics

    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_stage_begin(self):
        pass

    def on_stage_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        # haha, this case Callbacks merge it own class to itself
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def set_runner(self, runner):
        super().set_runner(runner)
        for callback in self.callbacks:
            callback.set_runner(runner)

    def on_batch_begin(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(i, **kwargs)

    def on_batch_end(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(i, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_stage_begin(self):
        for callback in self.callbacks:
            callback.on_stage_begin()

    def on_stage_end(self):
        for callback in self.callbacks:
            callback.on_stage_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class CheckpointSaver(Callback):
    def __init__(self, save_dir, save_name, num_checkpoints, mode, metric_name):
        super().__init__()
        self.mode = mode
        self.save_name = save_name
        # epoch_2.pth
        # num_checkpoints=4, why need  PriorityQueue ?
        # seem like it try to set size of  PriorityQueue
        self._best_checkpoints_queue = PriorityQueue(num_checkpoints)
        self._metric_name = metric_name
        # metric_name='DiceMetric',
        self.save_dir = save_dir
        # #/root/dumps/logs/resnet34_768_unet/fold0

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_stage_begin(self):
        # note, this inherit the set_runner method from class Callback
        os.makedirs(self.save_dir / self.runner.current_stage_name, exist_ok=True)
        while not self._best_checkpoints_queue.empty():
            self._best_checkpoints_queue.get()

    def save_checkpoint(self, epoch, path):
        if hasattr(self.runner.model, 'module'):
            state_dict = self.runner.model.module.state_dict()
        else:
            state_dict = self.runner.model.state_dict()
        torch.save(state_dict, path)

    def on_epoch_end(self, epoch):

        metric = self.metrics.val_metrics[self._metric_name]
        # metric = self.metrics.functions[self._metric_name]
       
        new_path_to_save = os.path.join(
            self.save_dir / self.runner.current_stage_name,
            # this seem weird, will check later 
            self.save_name.format(epoch=epoch, metric='{:.5}'.format(metric))
        )

        if self._try_update_best_losses(metric, new_path_to_save):
            self.save_checkpoint(epoch=epoch, path=new_path_to_save)

    def _try_update_best_losses(self, metric, new_path_to_save):
        # metric itself is an order  to priority queue
        if self.mode == 'min':
            metric = -metric
        if not self._best_checkpoints_queue.full():
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True
        # does get will get the lowest of priority ?
        min_metric, min_metric_path = self._best_checkpoints_queue.get()

        if min_metric < metric:
            os.remove(min_metric_path)
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        self._best_checkpoints_queue.put((min_metric, min_metric_path))
        return False


class TensorBoard(Callback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
         #/root/dumps/logs/resnet34_768_unet/fold0
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics.train_metrics.items():
            self.writer.add_scalar(f'train/{k}', float(v), global_step=epoch)

        for k, v in self.metrics.val_metrics.items():
            self.writer.add_scalar(f'val/{k}', float(v), global_step=epoch)

        # this one should note more
        for idx, param_group in enumerate(self.runner.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'group{idx}/lr', float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()


#  what is purpose of logger ?
class Logger(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.logger = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._get_logger(str(self.log_dir / 'logs.txt'))
        self.logger.info(f'Starting training with params:\n{self.runner.factory.params}\n\n')

    def on_epoch_begin(self, epoch):
        self.logger.info(
            f'Epoch {epoch} | '
            f'optimizer "{self.runner.optimizer.__class__.__name__}" | '
            f'lr {self.current_lr}'
        )

    def on_epoch_end(self, epoch):
        self.logger.info('Train metrics: ' + self._get_metrics_string(self.metrics.train_metrics))
        self.logger.info('Valid metrics: ' + self._get_metrics_string(self.metrics.val_metrics) + '\n')

    def on_stage_begin(self):
        self.logger.info(f'Starting stage:\n{self.runner.current_stage}\n')

    @staticmethod
    def _get_logger(log_path):
        # this will log in to the "log_path"
        # can ask chatGPT for more detail if forget
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def current_lr(self):
        res = []
        for param_group in self.runner.optimizer.param_groups:
            res.append(param_group['lr'])
        if len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def _get_metrics_string(metrics):
        # what are example
        return ' | '.join('{}: {:.5f}'.format(k, v) for k, v in metrics.items())
