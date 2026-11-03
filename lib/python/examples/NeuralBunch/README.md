# Neural bunch initializer (PICMI example)

Minimal PICMI setup that loads an externally generated openPMD electron bunch
and adds its fields and particles at the start of a PIConGPU simulation.

The example uses the deterministic generator from
`tools/neural-bunch/generator` and the checked-in plumbing artifact
`tools/neural-bunch/training/smoke/model`. The smoke artifact is **not** a
trained physics model; the default `referenceOnly` policy never reads it and
uses the exact FP64 reference kernel instead.

## Prerequisites

```bash
# PIConGPU Python package (from the repository root)
pip install -e "lib/python[test]"

# generator dependencies (openpmd-api, numpy, h5py; ADIOS2 for .bp5)
pip install openpmd-api h5py numpy
```

Build PIConGPU with the feature enabled:

```bash
pic-configure -DPIC_USE_neuralBunch=ON /path/to/build
```

## Run

From this directory:

```bash
# HDF5, exact reference fields, 2 steps
python main.py --backend hdf5 --field-initializer referenceOnly --steps 2

# ADIOS2/BP5
python main.py --backend adios2 --field-initializer referenceOnly --steps 2

# Neural model inside the artifact domain
python main.py --backend hdf5 --field-initializer neural --steps 2
```

`main.py` calls the generator for you when the series is missing. The exact
generator call is:

```bash
python ../../../../tools/neural-bunch/generator/generate_bunch.py \
    --output neural_bunch_data/hdf5/bunch --backend hdf5 \
    --iteration 0 --species e --model stationary --seed 1001 \
    --n 16 --grid 16 16 8 --cell-size 1e-7 1e-7 1e-7 \
    --centroid 8 8 4 --sigma 2 2 2
```

To generate the setup without running:

```bash
python main.py --backend hdf5 --generate-only
python -c 'import main; main.build_simulation("hdf5", "referenceOnly", 2).write_input_file("neural_bunch_data/hdf5/setup")'
```

Then the standard PIConGPU workflow:

```bash
pic-create neural_bunch_data/hdf5/setup myBunch
cd myBunch
pic-build -DPIC_USE_neuralBunch=ON
tbg -s bash -c etc/picongpu/N.cfg -t etc/picongpu/bash/mpiexec.tpl .
```

A single rank for two steps:

```bash
mpirun -n 1 bin/picongpu -d 1 1 1 -g 16 16 8 -s 2 --periodic 1 1 1
```

## What the PICMI object controls

```python
picmi.NeuralBunchInitializer(
    filename=".../bunch.h5",       # openPMD series, HDF5 or ADIOS2/BP5
    iteration=0,                    # iteration inside the series
    source_species="e",             # species name inside the series
    model_path=".../model",         # <model_path>.json + .bin
    chunk_size=1 << 20,             # streamed source particles per chunk
    tile_batch=1,                   # tiles per inference batch
    field_initializer="referenceOnly",  # neural | validatedNeural | referenceOnly
    enable_reference_fallback=True,     # exact reference outside the model domain
    time_policy=0,                      # source time == injection time
    species=electrons,                  # destination electron species
)
```

Attach it with `picmi.Simulation(..., picongpu_neural_bunch=initializer)`.
`dx`/`dy`/`dz`, the Yee field layout and the destination species alias are
derived from the grid and solver. The destination species must be an electron
species with the TSC (`"quadratic"`) shape and must not have an
`initial_distribution`, because the file supplies all particles.
