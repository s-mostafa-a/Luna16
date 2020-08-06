# LUNA16 object detection
In this repository, I have tried to develop a readable documented code for Lung Nodule Detection. (Based on the approach of the team "[grt123](https://github.com/lfz/DSB2017)" which stood on the first place of [DSB2017](https://www.kaggle.com/c/data-science-bowl-2017/leaderboard))

Their code (except for their model) was not readable to me at all, hence I have gone through [their paper](https://arxiv.org/abs/1711.08324).
And it is normal for their repository not to be readable, because it is originally coded for a competition! So there is no blame on them. :)


Their code can not be used without GPU.
Also, another disadvantage of their code is that they use data pre-processing and augmentation at the training time, I have decoupled the preprocess and augmentation from training.

I hope it helps researchers.
If you have any questions on the code, please send an [email to me](mailto:s.mostafa.a96@gmail.com?subject=[GitHub]%20LUNA16%20grt123).
# Code description
## Prepare
I have written the code of preparation in the `prepare` directory, it contains almost all data pre-processing and augmentation steps of [their paper](https://arxiv.org/abs/1711.08324).

### Tutorials:
I have made a jupyter notebook for pre-processing [here](./notebooks/Preprocessor.ipynb)
It is like a tutorial which I have tried to cover all of the works done in the `prepare._ct_scan.CTScan.preprocess` method and the main reference (the paper).

Also for the data augmentation, there is another jupyter notebook [here](./notebooks/Augmentor.ipynb) which this one is tutorial-ish too. 
If you want to know how does the code generate augmented patches, I strongly recommend you to read it.

## Model
To understand the "Nodule Net", it would be best if you read the paper. 
But for taking a brief look at the network structure, you can look at the below image.
Its code in `model/net.py` is a copy-paste of the grt123 code with minor changes.
Also, loss computation at `model/loss.py` is an IOU approach, to know the details you can read their paper.

![Net](./notebooks/figs/net.png)

## Main
In `main/dataset.py` I have written the `LunaDataSet` class which loads the saved augmented data to a torch `Dataset` and uses it to form a `DataLoader` and then feed the model as well as computing the loss.

# How to use
1. Download the dataset from [here](http://academictorrents.com/collection/luna-lung-nodule-analysis-16---isbi-2016-challenge).
**If you are just testing this code, it may be better to download just one subset of it because the volume of the data is too high to download. You should download CSV files too.**
Also, for more information, the dataset description is available [here](https://luna16.grand-challenge.org/data/).
2. Change the first 3 variables in `configs.py` file

3. Run `prepare/run.py`

4. Run `main/train.py`
