Instructions for updating the documentation and hosting on open source github repo.

On the same level as your Neuromancer repo, make a docs directory 
and then clone the gh-pages branch of the repo at github.com/pnnl/neuromancer

```bash

$ mkdir docs; cd docs
$ git clone https://github.com/pnnl/neuromancer.git -b

```

You will want to install sphinx in order to autogenerate the documentation

```bash

$ conda activate neuromancer
$ conda install sphinx
$ conda install -c conda-forge sphinx_rtd_theme

```

conda install -c conda-forge sphinx_rtd_theme
Now navigate to the docs folder in neuromancer and run the makefile to generate docs. 


```bash

$ cd ../neuromancer/docs
$ make html

```

Now navigate to the gh-pages branch you cloned and replace old html with newly generated html files. 
Take a look at the generated documentation by loading index.html in your browser. 
If everything looks good then add, commit, and push to the repo. 

```bash

$ cd ../../docs/neuromancer/html
$ mv *.html ../; cd ../; rm -rf html
$ git add *.html; git add objects.inv; git add search_index.js
$ git commit -m 'Added new documentation for NM version x.xx'
$ git push origin gh-pages

```