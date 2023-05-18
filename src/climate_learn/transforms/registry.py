TRANSFORMS_REGISTRY = {}


def register(name):
    def decorator(transform_cls):
        TRANSFORMS_REGISTRY[name] = transform_cls
        return transform_cls

    return decorator
