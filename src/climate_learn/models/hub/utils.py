MODEL_REGISTRY = {}


def register(name):
    def decorator(model_class):
        MODEL_REGISTRY[name] = model_class
        return model_class

    return decorator
