from climate_learn.data_module.tasks.args import TaskArgs


class DownscalingArgs(TaskArgs):
    task_class = "Downscaling"

    def __init__(
        self,
        dataset_args,
        highres_dataset_args,
        in_vars,
        out_vars,
        constant_names=[],
        subsample=1,
        split="train",
    ):
        super().__init__(
            dataset_args, in_vars, constant_names, out_vars, subsample, split
        )
        self.highres_dataset_args = highres_dataset_args

    def setup(self, data_module_args):
        super().setup(data_module_args)
        self.highres_dataset_args.split = self.split
        self.highres_dataset_args.setup(data_module_args)
