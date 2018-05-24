# Working Memory Networks
Code to reproduce the results from the paper [Working Memory Networks: Augmenting Memory Networks with a Relational Reasoning Module]() accepted as long paper at ACL 2018.

```
@inproceedings{pavez_2018_ACL,
  title={Working Memory Networks: Augmenting Memory Networks with a Relational Reasoning Module},
  author={Juan Pavez, H\'ector Allende, H\'ector Allende-Cid},
  booktitle={ACL},
  year={2018}
}
```
The Working Memory Network is a [Memory Network](https://arxiv.org/abs/1503.08895) architecture with a novel working memory storage and relational reasoning module.
The model retains the relational reasoning abilities of the [Relation Network](https://arxiv.org/abs/1706.01427) while reducing its computational complexity considerably. The model achieves state-of-the-art performance in the jointly trained [bAbI-10k](https://arxiv.org/abs/1502.05698) dataset, with an average error of less than 0.5%.

![](/plots/paper/working_memory_networks.png)

## Instructions

### Prerequisites

The code uses Python 2.7, Keras v1.2.2 and Theano v1.0.0. Please be sure to have those versions in your system.
Start downloading the bAbI dataset in the folder `data/babi` and extract it:
- `wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz`
- `tar -xzvf tasks_1-20_v1-2.tar.gz`

For NLVR, download the nlvr dataset from [here](https://github.com/clic-lab/nlvr) and save it to `data/nlvr`.

### Running the code

To run the Working Memory Network on bAbI (`WMemNN_bAbI.py`):
- `python WMemNN_bAbI.py ez lz mx rd lr seed`

Where: 
- `ez`: Embedding size (int)
- `lz`: GRU hidden units (int)
- `mx`: Number of facts used (int)
- `rd`: Run code used when restarting training. (int)*
- `lr`: Learning rate (float)
- `seed`: Random seed (int)

The code saves the model in the `models` folder (be sure to have that folder).

To run the Working Memory Network on NLVR:
- `python WMemNN_NLVR.py ez cz lz mx seed`

Where
- `ez`: Embedding size (int)
- `cz`: CNN layer number of units (int)
- `lz`: GRU hidden units (int)
- `mx`: Number of facts used (int)
- `seed`: Random seed (int)

To run the Relation Network on bAbI:
- `python RN_bAbI.py`

*In some cases we found useful to restart training after 400 epochs with a much smaller learning rate of 1e-5. To do this you can run `python babi_working_memnn.py ez lz mx 1 1e-5 seed` using the same seed and configuration than the previous run.
