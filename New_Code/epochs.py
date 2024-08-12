'''
Define Training and Validation epochs. Log files of metrics and loss.
'''

from metrics import image_acc_metric, class_acc_metric
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm as tqdm
import sys
import torch

class Epoch_imlarge:

    def __init__(self, model, loss, metrics, stage_name, device=None, verbose=True, 
                 image_acc = image_acc_metric, class_acc = class_acc_metric, num_classes = 11):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.acc_per_image = image_acc
        self.acc_per_class = class_acc
        self.num_classes = num_classes # 10 wound classes + 1 background class
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def batch_update(self, x, y, w):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = smp.utils.meter.AverageValueMeter()
        metrics_meters = {metric.__name__: smp.utils.meter.AverageValueMeter() for metric in self.metrics}
        class_accs = np.full((1,11), np.nan) # nan because nan values are ignored in the calculation of the mean accuracy 
        class_correct = np.full((1,11), np.nan)
        class_total = np.full((1,11), np.nan)
        
        with tqdm(dataloader, ncols=180, desc=self.stage_name, file=sys.stdout, 
                  position = 0, leave= True, disable=not (self.verbose)) as iterator:
            for count, (x, x_large, y_true, weights) in enumerate(iterator):
                    
                x, x_large, y_true, weights = x.to(self.device), x_large, y_true.to(self.device), weights.to(self.device)
                
                loss, y_pred = self.batch_update(x, x_large, y_true, weights)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {"loss" : loss_meter.mean.round(3)}
                logs.update(loss_logs)

                # update metrics logs

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred[:,1:,:,:], y_true).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: round(v.mean, 3) for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
            
                b, w, h = y_true.size()[0], y_true.size()[2], y_true.size()[3]

                y_true = torch.cat((torch.full((b,1,w,h), 0.8).to(self.device), y_true), dim=1) 
                # 0.8 makes sure that the background is only argmax if there is no wound with gbce=1
                # value 0.8 is a random value >0 and <1
                
                # For each epoch the pixel accuracies of the wounds per batch are collected
                # -> small wound areas gain more influence on the final pixel accuracy
                class_accs = np.concatenate((class_accs, 
                        self.acc_per_image(y_pred, y_true, self.num_classes)[None,:]), axis = 0)
                
                # For each epoch the number of all correct pixel classifications per class are collected
                # Additionally, for each epoch the number of all pixels per class are collected
                cor, tot = self.acc_per_class(y_pred, y_true, self.num_classes)
                class_correct = np.concatenate((class_correct, cor[None,:])) 
                class_total = np.concatenate((class_total, tot[None,:])) 

                logs.update({"acc" : list(np.round(np.nansum(class_correct[1:], axis=0)/np.nansum(class_total[1:], axis=0), 3))})
                logs.update({"mean_acc_per_image" : list(np.round(np.nanmean(class_accs[1:], axis = 0), 3))})
                
                if self.verbose:
                    iterator.set_postfix_str(logs)

        return logs


class TrainEpoch_imlarge(Epoch_imlarge):

    def __init__(self, model, loss, metrics, optimizer, device=None, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, x_large, y, w):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x, x_large)

        loss = self.loss(prediction[:,1:,:,:], y, w)

        loss.backward()

        self.optimizer.step()
            
        return loss, prediction

class ValidEpoch_imlarge(Epoch_imlarge):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, x_large, y, w):
        with torch.no_grad():
            prediction = self.model.forward(x, x_large)
            loss = self.loss(prediction[:,1:,:,:], y, w)

        return loss, prediction
    
class Epoch:

    def __init__(self, model, loss, stage_name, device=None, verbose=True, 
                 image_acc = image_acc_metric, class_acc = class_acc_metric, num_classes = 2):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.acc_per_image = image_acc
        self.acc_per_class = class_acc
        self.num_classes = num_classes # 10 wound classes + 1 background class
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def batch_update(self, x, y, w):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}

        class_accs = np.full((1, self.num_classes), np.nan) # nan because nan values are ignored in the calculation of the mean accuracy 
        class_correct = np.full((1, self.num_classes), np.nan)
        class_total = np.full((1, self.num_classes), np.nan)
        losses = []
        
        with tqdm(dataloader, ncols=250, desc=self.stage_name, file=sys.stdout, 
                  position = 0, leave= True, disable=not (self.verbose)) as iterator:
            for count, (x, gbce) in enumerate(iterator):
                    
                x, gbce = x.to(self.device), gbce.to(self.device)
                
                loss, y_pred = self.batch_update(x, gbce)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                losses.append(loss_value)
                # loss_meter.add(loss_value)
                loss_logs = {"loss" : np.mean(losses).round(3)}
                logs.update(loss_logs)
            
                b, w, h = gbce.size()[0], gbce.size()[2], gbce.size()[3]

                # gbce = torch.cat((torch.full((b,1,w,h), 0.8).to(self.device), gbce), dim=1) 
                # 0.8 makes sure that the background is only argmax if there is no wound with gbce=1
                # value 0.8 is a random value >0 and <1
                
                # For each epoch the pixel accuracies of the wounds per batch are collected
                # -> small wound areas gain more influence on the final pixel accuracy
                class_accs = np.concatenate((class_accs, 
                        self.acc_per_image(y_pred, gbce, self.num_classes)[None,:]), axis = 0)
                
                # For each epoch the number of all correct pixel classifications per class are collected
                # Additionally, for each epoch the number of all pixels per class are collected
                cor, tot = self.acc_per_class(y_pred, gbce, self.num_classes)
                class_correct = np.concatenate((class_correct, cor[None,:])) 
                class_total = np.concatenate((class_total, tot[None,:])) 

                logs.update({"acc" : list(np.round(np.nansum(class_correct[1:], axis=0)/np.nansum(class_total[1:], axis=0), 3))})
                logs.update({"mean_acc_per_image" : list(np.round(np.nanmean(class_accs[1:], axis = 0), 3))})
                
                if self.verbose:
                    iterator.set_postfix_str(logs)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, optimizer, device=None, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)

        loss = self.loss(prediction[:,:,:,:], y)

        loss.backward()

        self.optimizer.step()
            
        return loss, prediction

class ValidEpoch(Epoch):

    def __init__(self, model, loss, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction[:,:,:,:], y)

        return loss, prediction