#!/usr/bin/env bash

seq 1 1000 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --output /home/m.domrachev/repos/TextWorld/experiments/games/train/ --seed