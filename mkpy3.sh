#!/bin/bash
set -e
if [ "$(git symbolic-ref --short -q HEAD)" = "master" ]; then
    git checkout py3
    git checkout master -- .
    git commit -a -m "update from master"
    2to3 -W -n bct test setup.py
    git commit -a -m "run 2to3"
    nosetests --ignore-files="very_long_tests|nbs_tests"
    git checkout master
fi
