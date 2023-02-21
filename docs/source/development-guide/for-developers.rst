For Developers
==============


Installation
------------

Install ClimateLearn from source:

.. code-block:: console
    
    git clone https://github.com/aditya-grover/climate-learn.git
    pip install -e climate-learn

Development and documentation dependencies can be installed by specifying the extras :code:`[dev]` and :code:`[docs]`, respectively.


Formatting and Linting
----------------------
ClimateLearn uses `black <https://black.readthedocs.io/en/stable/>`_ for formatting and `flake8 <https://flake8.pycqa.org/en/latest/>`_ for linting.

.. code-block:: console

    black .
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics


Testing
-------
ClimateLearn uses `pytest <https://docs.pytest.org/>`_ for testing. Tests can be run as:

.. code-block:: console

    pytest tests/