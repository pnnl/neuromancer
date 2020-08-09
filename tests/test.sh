#!/bin/bash

for f in ../neuromancer/*.py; do echo $f; python $f; done
