import climate_learn as cl

# Sanity check to make sure that the registries are populated, which are used
# for the convenience loader methods. Does not check for specific entries in
# the registries.

def test_registries():
    assert len(cl.MODEL_REGISTRY) > 0
    assert len(cl.TRANSFORMS_REGISTRY) > 0
    assert len(cl.METRICS_REGISTRY) > 0