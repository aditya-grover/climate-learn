METRICS_REGISTRY = {}    

def register(name):
    def decorator(metric_class):
        METRICS_REGISTRY[name] = metric_class
        return metric_class
    return decorator