import copy
from collections import deque
import numpy as np
import prune.swa_utils as swa_utils


class SWADBase:
    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        raise NotImplementedError()

    def get_final_model(self):
        raise NotImplementedError()


class IIDMax(SWADBase):
    """SWAD start from iid max acc and select last by iid max swa acc"""

    def __init__(self, n_converge, n_tolerance, tolerance_ratio, validate, **kwargs):
        self.iid_max_acc = 0.0
        self.swa_max_acc = 0.0
        self.validate = validate
        self.avgmodel = None
        self.final_model = None

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn, args, val_loader, device, criterion, epoch, train_writer=None):

        if self.iid_max_acc < val_acc:
            self.iid_max_acc = val_acc
            self.avgmodel = swa_utils.AveragedModel(segment_swa.module, rm_optimizer=True)
            self.avgmodel.start_step = segment_swa.start_step

        self.avgmodel.update_parameters(segment_swa.module)
        self.avgmodel.end_step = segment_swa.end_step
        # (args, val_loader, self.avgmodel.module, device, criterion, epoch, train_writer=train_writer
        # evaluate
        print("inter validate")
        val_acc, _ = self.validate(args, val_loader, self.avgmodel.module, device, criterion, epoch, train_writer=train_writer)

        swa_val_acc = val_acc
        if swa_val_acc > self.swa_max_acc:
            self.swa_max_acc = swa_val_acc
            self.final_model = copy.deepcopy(self.avgmodel)

    def get_final_model(self):
        return self.final_model


class Nearbest(SWADBase):
    """SWAD start from iid max acc and select last by iid max swa acc"""

    def __init__(self, n_converge, n_tolerance, tolerance_ratio, validate, **kwargs):
        self.m1 = None
        self.m2 = None
        self.m3 = None
        self.fm1 = None
        self.fm2 = None
        self.fm3 = None
        self.max_acc = 0.0

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):

        segment_swa.acc = val_acc
        if self.m1 is None:
            self.m1 = segment_swa
            return 
        if self.m2 is None:
            self.m2 = segment_swa
            return 
        if self.m3 is None:
            self.m3 = segment_swa
            self.max_acc = self.m2.acc
            return 
        self.m1 = self.m2
        self.m2 = self.m3
        self.m3 = segment_swa
        if self.m2.acc > self.max_acc:
            self.max_acc = self.m2.acc
            self.fm2 = self.m2
            self.fm3 = self.m3
            self.fm1 = self.m1


    def get_final_model(self):
        final_model = swa_utils.AveragedModel(self.fm1.module)
        final_model.update_parameters(self.fm1.module)
        final_model.update_parameters(self.fm2.module)
        final_model.update_parameters(self.fm3.module)
        return final_model
    

class average_within_opech(SWADBase):
    """IIDMax has a potential problem that bias to validation dataset.
    LossValley choose SWAD range by detecting loss valley.
    """

    def __init__(self, n_converge, n_tolerance, tolerance_ratio, **kwargs):
        """
        Args:
            evaluator
            n_converge: converge detector window size.
            n_tolerance: loss min smoothing window size
            tolerance_ratio: decision ratio for dead loss valley
        """
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio

        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)

        self.final_model = None

        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    # @property
    # def is_converged(self):
    #     return self.converge_step is not None

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):

        if self.dead_valley:
            return

        frozen = copy.deepcopy(segment_swa.cpu())
        frozen.end_loss = val_loss
        # self.converge_Q.append(frozen)
        # self.smooth_Q.append(frozen)

        if self.final_model is None:
            self.final_model = swa_utils.AveragedModel(frozen)
        self.final_model.update_parameters(frozen, start_step=frozen.start_step, end_step=frozen.end_step)


    def get_final_model(self):
        return self.final_model.cuda()


class LossValley(SWADBase):
    """IIDMax has a potential problem that bias to validation dataset.
    LossValley choose SWAD range by detecting loss valley.
    """

    def __init__(self, n_converge, n_tolerance, tolerance_ratio, **kwargs):
        """
        Args:
            evaluator
            n_converge: converge detector window size.
            n_tolerance: loss min smoothing window size
            tolerance_ratio: decision ratio for dead loss valley
        """
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio

        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)

        self.final_model = None

        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    @property
    def is_converged(self):
        return self.converge_step is not None

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn=None):

        if self.dead_valley:
            return

        frozen = copy.deepcopy(segment_swa.cpu())
        frozen.end_loss = val_loss
        self.converge_Q.append(frozen)
        self.smooth_Q.append(frozen)

        if not self.is_converged:
            if len(self.converge_Q) < self.n_converge:
                return

            min_idx = np.argmin([model.end_loss for model in self.converge_Q])
            untilmin_segment_swa = self.converge_Q[min_idx]  # until-min segment swa.


            if min_idx == 0:
            # lossValley has been found
                self.converge_step = self.converge_Q[0].end_step
                self.final_model = swa_utils.AveragedModel(untilmin_segment_swa)
                # self.get_after_valley_point = 0

                th_base = np.mean([model.end_loss for model in self.converge_Q])
                self.threshold = th_base * (1.0 + self.tolerance_ratio)

                if self.n_tolerance < self.n_converge:
                    for i in range(self.n_converge - self.n_tolerance):
                        model = self.converge_Q[1 + i]
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                elif self.n_tolerance > self.n_converge:
                    converge_idx = self.n_tolerance - self.n_converge
                    Q = list(self.smooth_Q)[: converge_idx + 1]
                    start_idx = 0
                    for i in reversed(range(len(Q))):
                        model = Q[i]
                        if model.end_loss > self.threshold:
                            start_idx = i + 1
                            break
                    for model in Q[start_idx + 1 :]:
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                print(
                    f"Model converged at step {self.converge_step}, "
                    f"Start step = {self.final_model.start_step}; "
                    f"Threshold = {self.threshold:.6f}, "
                )
            return

        if self.smooth_Q[0].end_step < self.converge_step:
            return

        # converged -> loss valley
        min_vloss = self.get_smooth_loss(0)
        if min_vloss > self.threshold:
            self.dead_valley = True
            print(f"Valley is dead at step {self.final_model.end_step}")
            return

        model = self.smooth_Q[0]
        self.final_model.update_parameters(
            model, start_step=model.start_step, end_step=model.end_step
        )
        # self.get_after_valley_point += 1
        # if self.get_after_valley_point == 3:
        #     self.dead_valley = True
        #     print(f"Valley is dead at step {self.final_model.end_step}")


    def get_final_model(self):
        if not self.is_converged:
            print("Requested final model, but model is not yet converged; return last model instead")
            return self.converge_Q[-1].cuda()

        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                smooth_loss = self.get_smooth_loss(0)
                if smooth_loss > self.threshold:
                    break
                segment_swa = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment_swa, step=segment_swa.end_step)

        return self.final_model.cuda()
    