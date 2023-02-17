from climate_learn.data_module.tasks.args import TaskArgs


class ForecastingArgs(TaskArgs):
    task_class = "Forecasting"

    def __init__(
        self,
        dataset_args,
        in_vars,
        out_vars,
        constant_names=[],
        history: int = 1,
        window: int = 6,
        pred_range=6,
        subsample=1,
        split="train",
    ):
        super().__init__(
            dataset_args, in_vars, out_vars, constant_names, subsample, split
        )
        self.history = history
        self.window = window
        self.pred_range = pred_range

    def setup(self, data_module_args):
        super().setup(data_module_args)
