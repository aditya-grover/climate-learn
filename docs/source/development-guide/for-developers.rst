.. role:: python(code)
  :language: python
  :class: highlight

For Developers
==============

Installing from Source
----------------------

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

Notes about writing tests:

    #. Import the package as :python:`import climate_learn` rather than :python:`import src.climate_learn`.
    #. For tests that require access to resources not available on `GitHub Actions Runner <https://github.com/actions/runner>`_, use the following:

    .. code-block:: python

        # place this at the start of your test script
        GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

        # place this on the line before your test function definition
        @pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
        def my_local_test(...):
            ... # test function body

Documentation
-------------
ClimateLearn uses `reStructuredText <https://docutils.sourceforge.io/rst.html>`__. Example docstring:

.. code-block:: python
    :linenos:

    def doc_example_function(arg1, arg2, kwarg1=False):
        r"""One sentence description of the function.
            Use the following lines to talk about what the function does at a
            high level and to provide any necessary additional context.

        .. highlight:: python

        :param arg1: Description of the first argument.
        :type arg1: str
        :param arg2: Description of the second argument.
        :type arg2: int|float
        :param kwarg1: Description of the first keyword argument. Maybe we want
            to use some syntax highlighting here. This can be achieved as such:
            :python:`class MyClass`. Defaults to `False`.
        :type kwarg1: bool, optional
        :returns: Description of what the function returns.
        :rtype: bool
        """
        return False

Suppose this function is available at the top level of the :python:`climate_learn` package. It can be imported to the docs as such:

.. code-block:: rst

    .. This is a reStructuredText comment. Maybe this file is at docs/source/file.rst
    .. autofunction:: climate_learn.doc_example_function

Further readings:

    - `reStructuredText primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__
    - `Generating docs from docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`__
    - `ReadTheDocs tutorial <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`__
