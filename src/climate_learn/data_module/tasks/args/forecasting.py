from climate_learn.data_module.tasks.args import TaskArgs
from climate_learn.data_module.tasks.forecasting import Forecasting

class ForecastingArgs(TaskArgs):
    task_class = Forecasting

    def __init__(
        self,
        dataset_args,
        in_vars,
        constant_names,
        out_vars,
        history,
        window,
        pred_range,
        subsample=1,
        split="train",
    ):
        super.__init__(dataset_args, in_vars, constant_names, out_vars, subsample, split)
        self.history = history
        self.window = window
        self.pred_range = pred_range

    def setup(self, data_module_args):
        super().setup(data_module_args)