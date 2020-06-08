#!/bin/bash

cd ../deepmpc/

for f in *.py; do echo $f; python $f; done