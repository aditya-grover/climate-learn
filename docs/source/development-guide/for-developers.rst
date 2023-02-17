For Developers
==============


Installation
------------

Clone ClimateLearn from GitHub:

.. code-block:: console
    git clone https://github.com/aditya-grover/climate-learn.git

If you are interested in making changes to ClimateLearn's source code, install the optional ``dev`` dependencies:

.. code-block:: console

    pip install -e .[dev]

If you are interested in making changes to ClimateLearn's documentation, install the optional ``docs`` dependencies:

.. code-block:: console

    pip install -e .[docs]

These two commands can be combined into one:

.. code-block:: console

    pip install -e .[dev,docs]


Formatting and Linting
----------------------
ClimateLearn uses `black <https://black.readthedocs.io/en/stable/>`_ for formatting and `flake8 <https://flake8.pycqa.org/en/latest/>`_ for linting.


Testing
-------
ClimateLearn uses `pytest <https://docs.pytest.org/>`_ for testing. 