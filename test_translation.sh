#!/bin/bash

set -e
pushd ./checkpoints/save/
cat checkpoint_20250402_150847_40_part_aa checkpoint_20250402_150847_40_part_ab > checkpoint_20250402_150847_40.bin
popd
./transformer -e 0 -c ./checkpoints/save/checkpoint_20250402_150847_40.bin
