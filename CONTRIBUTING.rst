Contributing
============

The recommended procedure for contributing code to this repository is detailed
here. It is the standard method of contributing to Github based repositories
(https://help.github.com/articles/fork-a-repo).

If you have don't have access rights to this repository then you should fork
the repository on Github using the Github UI and clone the fork that you just
made to your machine::

   git clone git@github.com:<your-username>/GaitAnalysisToolKit.git

Change into the directory::

   cd GaitAnalysisToolKit

Now, setup a remote called ``upstream`` that points to the main repository so
that you can keep your local repository up-to-date::

   git remote add upstream git@github.com:csu-hmc/GaitAnalysisToolKit.git

Now you have a remote called ``origin`` (the default) which points to **your**
Github account's copy and a remote called ``upstream`` that points to the main
repository on the csu-hmc organization Github account.

It's best to keep your local master branch up-to-date with the upstream master
branch and then branch locally to create new features. To update your local
master branch simply::

   git checkout master
   git pull upstream master

If you have access rights to the main repository simply, clone it and don't
worry about making a fork on your Github account::

   git clone git@github.com:csu-hmc/GaitAnalysisToolKit.git

Change into the directory::

   cd GaitAnalysisToolKit

Now, to contribute a change to the repository you should create a new branch
off of the local master branch::

   git checkout -b my-branch

Now make changes to the software and be sure to always include tests! Make sure
all tests pass on your machine with::

   nosetests

Once tests pass, add any new files you created::

   git add my_new_file.py

Now commit your changes::

   git commit -am "Added an amazing new feature."

Push your commits to a mirrored branch on the Github repository that you
cloned::

   git push origin my-branch

Now visit the repository on Github (either yours or the main one) and you
should see a "compare and pull button" to make a pull request against the main
repository. Github and Travis-CI will check for merge conflicts and run the
tests again on a cloud machine. You can ask others to review your code at this
point and if all is well, press the "merge" button on the pull request.
Finally, delete the branches on your local machine and on your Github repo::

   git branch -d my-branch && git push origin :my-branch

Git Notes
---------

- The master branch on main repository on Github should always pass all tests
  and we should strive to keep it in a stable state. It is best to not merge
  contributions into master unless tests are passing, and preferably if
  someone else approved your code.
- In general, do not commit changes to your local master branch, always pull in
  the latest changes from the master branch with ``git pull upstream master``
  then checkout a new branch for your changes. This way you keep your local
  master branch up-to-date with the main master branch on Github.
- In general, do not push changes to the main repo master branch directly, use
  branches and push the branches up with a pull request.
- In general, do not commit binary files, files generated from source, or large
  data files to the repository. See
  https://help.github.com/articles/working-with-large-files for some reasons.
