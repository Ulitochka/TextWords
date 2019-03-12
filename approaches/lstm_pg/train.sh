#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

python3 ${SCRIPT_PATH}/train.py /home/m.domrachev/repos/TextWorld/TextWord_starting_kit/sample_games/