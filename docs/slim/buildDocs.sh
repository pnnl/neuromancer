cat > buildDocs.sh <<'EEOOFF'
#!/bin/bash
set -x
################################################################################
# File:    buildDocs.sh
# Purpose: Script that builds our documentation using sphinx and updates GitHub
#          Pages. This script is executed by:
#            .github/workflows/docs_pages_workflow.yml
#
# Authors: Michael Altfield <michael@michaelaltfield.net>
# Created: 2020-07-17
# Updated: 2020-07-17
# Version: 0.1
################################################################################

###################
# INSTALL DEPENDS #
###################

apt-get update
apt-get -y install git rsync python3-sphinx python3-sphinx-rtd-theme

#####################
# DECLARE VARIABLES #
#####################

pwd
ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

##############
# BUILD DOCS #
##############

# build our documentation with sphinx (see docs/conf.py)
# * https://www.sphinx-doc.org/en/master/usage/quickstart.html#running-the-build
make -C docs clean
make -C docs html

#######################
# Update GitHub Pages #
#######################

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

docroot=`mktemp -d`
rsync -av "docs/_build/html/" "${docroot}/"

pushd "${docroot}"

# don't bother maintaining history; just generate fresh
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# add .nojekyll to the root so that github won't 404 on content added to dirs
# that start with an underscore (_), such as our "_content" dir..
touch .nojekyll

# Add README
cat > README.md <<EOF
# GitHub Pages Cache

Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.


If you're looking to update our documentation, check the relevant development branch's 'docs/' dir.

For more information on how this documentation is built using Sphinx, Read the Docs, and GitHub Actions/Pages, see:

 * https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
EOF

# copy the resulting html pages built from sphinx above to our new git repo
git add .

# commit all the new files
msg="Updating Docs for commit ${GITHUB_SHA} made on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"

# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force

popd # return to main repo sandbox root

# exit cleanly
exit 0
EEOOFF
chmod +x buildDocs.sh