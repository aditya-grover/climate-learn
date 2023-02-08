For Maintainers
===============

Managing Issues
---------------
Coming soon!

Managing Pull Requests
----------------------

#. Comments and requests for changes should be polite and actionable.

#. Pull requests should be `atomic <https://en.wikipedia.org/wiki/Atomic_commit>`_. If not, ask the person who opened the request to break it down into multiple, smaller ones. 

#. Resolve conflicts via `rebase <https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase>`_.

#. After approving a pull request, it should be `"squashed and merged" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits>`_ with a descriptive, present-tense sentence as the title. Provide a description as appropriate.

Example of a good pull request: `"#33: Make Linear Regression baseline optional for testing" <https://github.com/aditya-grover/climate-learn/pull/33>`_.

* Title is a descriptive, present-tense sentence.
* Pull request is atomic: it is clear from the title that exactly one change is made.

Example of a bad pull request: `"#12: Uncertainty merge" <https://github.com/aditya-grover/climate-learn/pull/12>`_.

* Title is not sufficiently descriptive.
* Pull request is massive: many changes are introduced simultaneously. This should have been multiple smaller pull requests.