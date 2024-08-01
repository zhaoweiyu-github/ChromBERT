chrombert_make_dataset
**********************

Generate general datasets for ChromBERT from bed files.

.. code-block:: shell

    chrombert_make_datasets [OPTIONS] BED

.. rubric:: Options 

.. option:: BED

    Path to the bed file.

.. option:: -o, --oname

    Path to the output file. Stdout if not specified. Must end with .tsv or .txt.

.. option:: --mode

    Mode to generate the dataset. Choices are:
    
    - *region*: only consider overlap between input regions to determine the label generated. Useful for narrowPeak-like input.
    - *all*: report all overlapping status like bedtools intersect -wao. You should determine the label column by yourself.

    Default is *region*.

.. option:: --center

    If used, only consider the center of the input regions.

.. option:: --label

    If mode is not *region*, this column will be used as the label. Default is the 4th column (1-based).

.. option:: --no-filter

    Do not filter the regions that are not overlapped.

.. option:: --basedir

    Base directory for the required files. Default is set to the value of `DEFAULT_BASEDIR`.

.. option:: -g, --genome

    Genome version. For example, hg38 or mm10. Only hg38 is supported now. Default is *hg38*.

.. option:: -hr, --high-resolution

    Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.

