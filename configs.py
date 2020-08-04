# some directory for output the results:
OUTPUT_PATH = '/Users/mostafa/PycharmProjects/Luna/prepare/tmp'

# resource path which contains: annotations.csv, candidates.csv, and [subset0]
RESOURCES_PATH = '/Users/mostafa/Desktop/dsb_analyse/input'
# if you downloaded all subsets, put their files in one directory and change the following variable to its name
SUB_DIRECTORY = '/subset0/'



BLOCK_SIZE = 128
TARGET_SHAPE = (32, 32, 32, 3, 5)
COORDS_SHAPE = (3, 32, 32, 32)
ANCHOR_SIZES = [10, 30, 60]
VAL_PCT = 0.2
TOTAL_EPOCHS = 100
DEFAULT_LR = 0.01
