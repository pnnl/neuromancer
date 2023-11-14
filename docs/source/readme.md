#### Update: Automated Documentation Process 
Documentation will now be built and deployed to gh-pages branch (https://pnnl.github.io/neuromancer/) automatically on 
merge with master branch. This is done through a Github Actions (GHA) script found at ./github/workflows/update_docs.yml. 
No need to manually build on your local machine. 

For information on the update_docs.yml GHA script, please refer to the end of this readme.

For reference, the manual instructions for updating the documentation is shown below: 

#### Manual Process for Documentation Updates
On the same level as your Neuromancer repo, make a docs directory 
and then clone the gh-pages branch of the repo at github.com/pnnl/neuromancer

```bash

$ mkdir docs; cd docs
$ git clone https://github.com/pnnl/neuromancer.git -b gh-pages --single-branch
$ mv neuromancer html

```

You will want to install sphinx in order to autogenerate the documentation

```bash

$ conda activate neuromancer
$ conda install sphinx -c anaconda
$ conda install -c conda-forge sphinx_rtd_theme

```
Now navigate to the docs folder in neuromancer and run the makefile to generate docs. 


```bash

$ cd ../neuromancer/docs
$ make html

```

Now navigate to the gh-pages branch you cloned (now called html instead of neuromancer).  
Take a look at the generated documentation by loading index.html in your browser. 
If everything looks good then add, commit, and push to the repo. 

```bash

$ cd ../../docs/html
$ git add *.html; git add objects.inv; git add search_index.js
$ git commit -m 'Added new documentation for NM version x.xx'
$ git push origin gh-pages

```

#### Information on GitHub Actions
GitHub Actions is a continuous integration platform that allows for a set of actions to be executed
on any trigger event associated with any branch of a repository. GHA are defined in a yaml workflow (e.g)
update_docs.yml. This particular workflow, update_docs.yml, will run on any push to the master branch. 
GHA workflows are associated with a set of jobs. Each job is assigned a GitHub-hosted runner. A set of job(s) 
gets triggered based on the "on" clause at the top of of the workflow -- in this case a push to master. 
A job is defined as a set of steps. A step can be defined with a (uses, name, with, runs) block where uses defines the 
prerequisite steps needed for the runner to execute the associated runs(s). In this case we tell the 
runer to use the latest Ubuntu version. We also tell the runner to use python using the GHA syntactical sugar
"actions/setup-python@v3". It also uses "this" repo using the GHA syntactical sugar actions/checkout@v3. 
Step 3 in the workflow is equivalent to the install sphinx steps as outline in the manual process. Step 4
is equivalent to the "make html" command in the manual process. Step 5 leverages a GHA "extension" called 
actions-gh-pages (https://github.com/peaceiris/actions-gh-pages) to push the sphinx documentation to a 
gh-pages branch. The syntax used here is specific to actions-gh-pages. Before deploying to the master branch, 
we check that we are "on" the master branch. This statement might be obsolete due to the trigger event (push 
to master) defined at the top of this workflow; however, it is included due to recommended use within actions-gh-pages

