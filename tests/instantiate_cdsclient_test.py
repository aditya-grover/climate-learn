# Standard library
import os

# Third party
import cdsapi
import pytest

GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
def test_instantiate_cdsclient():
    c = cdsapi.Client()
