# openPMD restart metadata tests

The repository has no restart integration-test framework at this base. The
metadata validation is covered by a Catch2 unit test, and the end-to-end
restart behaviour by a script that needs a GPU and a prepared checkpoint.

## Unit test

`share/picongpu/unit/non_dimensional_tests/restartFieldMetadata.cpp` is picked
up by the existing `share/picongpu/unit` CMake project. It tests the conversion
factor, invalid `unitSI`, and the rejection of mismatched `gridSpacing`,
`position`, `axisLabels`, `unitDimension`, `timeOffset` and
`gridGlobalOffset`.

```bash
cmake -S share/picongpu/unit -B unit-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_DEP_CUDA=ON
ninja -C unit-build -j 16 'UnitTest-restartFieldMetadata.cpp'
./unit-build/non_dimensional_tests/UnitTest-restartFieldMetadata.cpp
```

## End-to-end test

`openpmd_restart_metadata_test.py` copies a checkpoint, mutates only its
openPMD metadata, and runs `picongpu --checkpoint.restart`. It checks:

1. an unmodified checkpoint restarts and produces the same fields,
2. a factor-2 `unitSI` is converted to internal units,
3. a mismatched `gridSpacing` is rejected with the record name and both values,
4. a checkpoint without `Convolutional PML` records restarts with periodic
   boundaries,
5. a malformed `picongpu_idProvider` mesh is rejected,
6. a non-`float_X` field dataset is rejected with the datatype in the error.

```bash
env -u LD_PRELOAD python3 test/restart/openpmd_restart_metadata_test.py \
  --picongpu /path/to/build/picongpu \
  --checkpoint /path/to/checkpoint_000000.h5 \
  --series checkpoint --step 0 \
  --setup /path/to/deck \
  --workdir /tmp/restart-metadata-test \
  --grid 16 16 8 --periodic 1 1 1 \
  --mpirun "mpirun --allow-run-as-root -n 1"
```

The checkpoint and the deck must agree on the field grid, cell size and time
step. The deck must provide `etc/openPMD_config_E.toml` and
`etc/openPMD_config_B.toml`; output is read from `openPMD/E_<step>.h5` and
`openPMD/B_<step>.h5`. `h5py` and NumPy are required.
