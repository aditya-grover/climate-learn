# Standard library
import os

# Third party
import cdsapi
import pytest


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", False),
                    reason="only works locally")
def test_instantiate_cdsclient():
    c = cdsapi.Client()
