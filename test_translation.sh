#!/bin/bash

set -e
cat checkpoint_20250402_150847_40_part_aa checkpoint_20250402_150847_40_part_ab > checkpoint_20250402_150847_40.bin
./transformer -e 0 -c ./checkpoint_20250402_150847_40.bin
