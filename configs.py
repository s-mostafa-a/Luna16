# some directory for output the results:
OUTPUT_PATH = '/Users/mostafa/PycharmProjects/Luna/prepare/tmp'

# Resource path which contains: annotations.csv, candidates.csv,
# and subdirectories containing .mhd files.
# This is the directory structure needed to run the code:
# (The code will use all .mhd and .raw files inside subdirectories which their name is in annotations or candidates)
'''
[RESOURCES_PATH]/
            annotations.csv
            candidates.csv
            subset0/
                        *.mhd
                        *.raw
            subset1/
                        *.mhd
                        *.raw
            my_custom_subset/
                        *.mhd
                        *.raw
'''
RESOURCES_PATH = '/Users/mostafa/Desktop/dsb_analyse/input'


PADDING_FOR_LOCALIZATION = 10
BLOCK_SIZE = 128
TARGET_SHAPE = (32, 32, 32, 3, 5)
COORDS_SHAPE = (3, 32, 32, 32)
ANCHOR_SIZES = [10, 30, 60]
VAL_PCT = 0.2
TOTAL_EPOCHS = 100
DEFAULT_LR = 0.01
