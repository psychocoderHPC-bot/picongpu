PIConGPU unit test
==================

Test components in an as best as possible isolated environment.

Example how to compile and execute the tests.compile

.. code-block:: bash
    # compile for NVIDIA GPUs
    cmake <path_to_picongpu_source>/share/picongpu/unit/ -Dalpaka_DEP_CUDA=ON
    make -j
    ctest
