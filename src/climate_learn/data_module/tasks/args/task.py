class TaskArgs:
    task_class = "Task"

    def __init__(
        self,
        dataset_args,
        in_vars,
        out_vars,
        constant_names=[],
        subsample=1,
        split="train",
    ):
        self.dataset_args = dataset_args
        self.in_vars = in_vars
        self.constant_names = constant_names
        self.out_vars = out_vars
        self.subsample = subsample
        self.split = split

    def setup(self, data_module_args):
        self.dataset_args.split = self.split
        self.dataset_args.setup(data_module_args)
