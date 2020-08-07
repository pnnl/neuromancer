#!/bin/bash

for f in ../deepmpc/*.py; do echo $f; python $f; done
