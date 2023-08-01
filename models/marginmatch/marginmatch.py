import pickle
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib
from train_utils import AverageMeter
from tqdm import tqdm

from .marginmatch_utils import AUMCalculator, consistency_loss, Get_Scalar
from train_utils import ce_loss, EMA, Bn_Controller
import random
from sklearn.metrics import *
from copy import deepcopy
from datasets.dataset import BasicDataset
from datasets.data_utils import get_data_loader

def replace_threshold_examples(ulb_dataset, aum_calculator):
  num_threshold_examples = min(len(ulb_dataset) // 100, len(ulb_dataset) // ulb_dataset.num_classes)
  threshold_data_ids = random.sample(list(range(len(ulb_dataset.data))), num_threshold_examples)
  ulb_dataset.switch_threshold_examples(threshold_data_ids)
  aum_calculator.switch_threshold_examples(threshold_data_ids)

class MarginMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class MarginMatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(MarginMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')


    def set_dset_ulb(self, ulb_dset):
        self.ulb_dset = ulb_dset

    def set_dset(self, train_dataset, eval_dataset):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset


    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train(self, args, logger=None):

        has_labels = True
        if 'stl' in args.dataset.lower():
            has_labels = False

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + \
            args.dataset + '_' + str(args.num_labels) + '.json'

        with open(dist_file_name, 'r') as f:
            p_target = json.loads(f.read())
            p_target = torch.tensor(p_target['distribution'])
            p_target = p_target.cuda(args.gpu)
        # print('p_target:', p_target)

        p_model = None

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label_confidence = torch.ones(
            (len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label_confidence = selected_label_confidence.cuda(args.gpu)
        classwise_acc_confidence = torch.zeros(
            (self.num_classes,)).cuda(args.gpu)

        aum_calculator = AUMCalculator(args.delta, int(
                self.num_classes), len(self.ulb_dset), args.percentile / 100)
        pbar = tqdm(total=2**20)
        threshold_example_cutoff = (args.num_train_iter * 9) // 10


        aum_threshold_steps = 0
        while True:
            delay = args.delay
            if self.it < threshold_example_cutoff and self.it >= args.warmup_period:
                replace_threshold_examples(self.ulb_dset, aum_calculator)
            else:
                self.ulb_dset.switch_threshold_examples(set())
                aum_calculator.switch_threshold_examples(set())
            
            dset_dict = {}
            dset_dict['train_ulb'] = self.ulb_dset

            self.loader_dict['train_ulb'] = get_data_loader(self.ulb_dset,
                                        args.batch_size * args.uratio,
                                        data_sampler=args.train_sampler,
                                        num_iters=args.num_train_iter,
                                        num_workers=4 * args.num_workers,
                                        distributed=args.distributed)
            if self.it == 0:
                aum_threshold = None

            self.it += 1
            for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s, y_ulb, threshold_mask) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

                if self.it % args.switcher_frequency == 0:
                    break

                pbar.update(1)
                if self.it % args.threshold_frequency_calculation_steps == 0 and self.it > args.warmup_period:
                    if delay < 0 and self.it < threshold_example_cutoff:
                        aum_threshold = aum_calculator.retrieve_threshold()
                elif self.it == args.warmup_period:
                    aum_threshold = 0
                    break
                elif self.it < args.warmup_period:
                    aum_threshold = None
                        
                delay -= 1

                y_ulb = y_ulb.cuda(args.gpu)
                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                num_lb = x_lb.shape[0]
                num_ulb = x_ulb_w.shape[0]

                assert num_ulb == x_ulb_s.shape[0]

                x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(
                    args.gpu), x_ulb_s.cuda(args.gpu)
                x_ulb_idx = x_ulb_idx.cuda(args.gpu)
                y_lb = y_lb.cuda(args.gpu)
                threshold_mask = threshold_mask.cuda(args.gpu)

                pseudo_counter_confidence = Counter(
                    selected_label_confidence.tolist())
                if max(pseudo_counter_confidence.values()) < len(self.ulb_dset):  # not all(5w) -1
                    if args.thresh_warmup:
                        for i in range(args.num_classes):
                            classwise_acc_confidence[i] = pseudo_counter_confidence[i] / \
                                max(pseudo_counter_confidence.values())
                    else:
                        wo_negative_one = deepcopy(pseudo_counter_confidence)
                        if -1 in wo_negative_one.keys():
                            wo_negative_one.pop(-1)
                        for i in range(args.num_classes):
                            classwise_acc_confidence[i] = pseudo_counter_confidence[i] / \
                                max(wo_negative_one.values())
                    classwise_acc_confidence[args.num_classes] = 0

                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

                # inference and calculate sup/unsup losses
                with amp_cm():
                    logits = self.model(inputs)
                    logits_x_lb = logits[:num_lb]
                    logits_x_ulb_w = logits[num_lb:(num_lb + num_ulb)]
                    logits_x_ulb_s = logits[(num_lb + num_ulb):]

                    sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                    # hyper-params for update
                    T = self.t_fn(self.it)
                    p_cutoff = self.p_fn(self.it)

                    unsup_loss, mask_confidence, mask_aum, select_confidence, _, pseudo_lb, p_model, conf_acc, both_acc, aum_acc, mask_sum = consistency_loss(logits_x_ulb_s,
                                                                                                                                        logits_x_ulb_w,
                                                                                                                                        classwise_acc_confidence,
                                                                                                                                        p_target,
                                                                                                                                        p_model,
                                                                                                                                        x_ulb_idx,
                                                                                                                                        aum_calculator,
                                                                                                                                        y_ulb,
                                                                                                                                        aum_threshold,
                                                                                                                                        self.num_classes,
                                                                                                                                        threshold_mask,
                                                                                                                                        'ce', T, p_cutoff,
                                                                                                                                        use_hard_labels=args.hard_label,
                                                                                                                                        use_DA=args.use_DA,
                                                                                                                                        labels=has_labels)

                    if x_ulb_idx[select_confidence == 1].nelement() != 0:
                        selected_label_confidence[x_ulb_idx[select_confidence == 1]
                                                ] = pseudo_lb[select_confidence == 1]

                    total_loss = sup_loss + self.lambda_u * unsup_loss


                # parameter updates
                if args.amp:
                    scaler.scale(total_loss).backward()
                    if (args.clip > 0):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if (args.clip > 0):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.clip)
                    self.optimizer.step()

                self.scheduler.step()
                self.ema.update()
                self.model.zero_grad()

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict = {}
                if aum_threshold is not None:
                    tb_dict['train/aum_threshold'] = aum_threshold
                tb_dict['train/sup_loss'] = sup_loss.detach()
                tb_dict['train/mask_sum'] = mask_sum.detach()
                tb_dict['train/unsup_loss'] = unsup_loss.detach()
                tb_dict['train/total_loss'] = total_loss.detach()

                tb_dict['train/mask_ratio_confidence'] = 1.0 - \
                    mask_confidence.detach()
                tb_dict['train/mask_ratio_aum'] = 1.0 - mask_aum.detach()
                tb_dict['train/mask_ratio_both'] = 1.0 - \
                    (mask_confidence * mask_aum).detach()

                if has_labels:
                    tb_dict['train/impurity_confidence'] = 1.0 - conf_acc.detach()
                    tb_dict['train/impurity_aum'] = 1.0 - aum_acc.detach()
                    tb_dict['train/impurity_both'] = 1.0 - both_acc.detach()

                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(
                    end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                if self.it % 20 == 0:
                    self.tb_log.update(tb_dict, self.it)


                # Save model for each 10K steps and best model for each 1K steps
                if self.it % 4000 == 0:
                    save_path = os.path.join(args.save_dir, args.save_name)
                    if not args.multiprocessing_distributed or \
                            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                        self.save_model('latest_model.pth', save_path)

                if self.it % self.num_eval_iter == 0:
                    print('Classwise confidence"', classwise_acc_confidence)
                    eval_dict = self.evaluate(args=args)
                    tb_dict.update(eval_dict)
                    save_path = os.path.join(args.save_dir, args.save_name)
                    if tb_dict['eval/top-1-acc'] > best_eval_acc:
                        best_eval_acc = tb_dict['eval/top-1-acc']
                        best_it = self.it
                    self.print_fn(
                        f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                    if not args.multiprocessing_distributed or \
                            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                        if self.it == best_it:
                            self.save_model('model_best.pth', save_path)
                        if not self.tb_log is None:
                            self.tb_log.update(tb_dict, self.it)

                self.it += 1
                del tb_dict
                start_batch.record()
                if self.it > 0.8 * args.num_train_iter:
                    self.num_eval_iter = 1000

            if self.it > args.num_train_iter:
                break
        eval_dict = self.evaluate(args=args)
        eval_dict.update(
            {'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def  evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)[:, :args.num_classes]
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]]
               for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
