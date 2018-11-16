#!/bin/bash

#awk -F $'\t'   '{print $8,$17,$18}' toy.csv > ptoy.csv
awk -F $'\t'   '{print $17,$18}' toy.csv > ptoy.csv

