from preprocess._ct_scan import CTScan
import pandas as pd
import matplotlib.pyplot as plt

test_filename = '1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'

annotations = pd.read_csv('/Users/mostafa/Desktop/dsb_analyse/input/annotations.csv')
candidates = pd.read_csv('/Users/mostafa/Desktop/dsb_analyse/input/candidates.csv')
nodule_coords = candidates[candidates['seriesuid'] == test_filename][candidates['class'] == 1]
tp_co = [(a['coordZ'], a['coordY'], a['coordX']) for a in nodule_coords.iloc]
ct = CTScan(filename=test_filename, coords=[tp_co[0]])
ct.transform()
previous = ct.get_subimages()
dicts_list = ct.get_augmented_subimages_around_coords()
for dikt in dicts_list:
    print(dikt['img'].shape, dikt['radius'], dikt['origin'], dikt['spacing'])
    plt.imshow(dikt['img'][dikt['origin'][0], :, :], cmap=plt.cm.gray)
    plt.savefig('./tmp/1.pdf')
    plt.show()
    plt.imshow(dikt['img'][:, dikt['origin'][1], :], cmap=plt.cm.gray)
    plt.savefig('./tmp/2.pdf')
    plt.show()
    plt.imshow(dikt['img'][:, :, dikt['origin'][2]], cmap=plt.cm.gray)
    plt.savefig('./tmp/3.pdf')
    plt.show()
    break

for im in previous:
    plt.imshow(im, cmap=plt.cm.gray)
    plt.savefig('./tmp/4.pdf')
    plt.show()
