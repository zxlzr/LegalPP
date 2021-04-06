# LegalPP

> Codes for paper "***Text-guided Legal Knowledge Graph Reasoning***"

- 使用BERT编码的嵌入作为图网络的节点嵌入
- 对R-GCN进行拓展，图编码器为：RGCN/GAT，关系解码器为DistMult/SimplE
- 对嵌入进行t-SNE可视化

## Project Structure
```python
.
├── config                      # model config files containing model combinations
│   ├── gat_dist.json
│   ├── gat_dist_pre_raw.json
│   ├── gat_dist_pre_tuned.json
│   ├── rgcn_dist.json
│   ├── rgcn_dist_pre_raw.json
│   ├── rgcn_dist_pre_tuned.json
│   ├── rgcn_simple.json
│   └── sample.json
├── data                        # dataset
│   ├── dev.tsv
│   ├── entities.txt
│   ├── entity2text.txt
│   ├── relation2text.txt
│   ├── relations.txt
│   ├── target                  # containing target relation triples: task-entry relation & filtered entity corrupt target
│   │   ├── filtered_entities.txt
│   │   └── test.tsv
│   ├── test.tsv
│   ├── title2entry.json
│   └── train.tsv
├── dataset.py
├── embeds
│   ├── *.npz                   # embedding created from bert model
│   ├── make_bert_embed.py      # embedding maker
│   ├── raw-bert                # raw bert model for creating embedding
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── tsne                    # result of t-SNE visualization
│   │   └── *.png
│   ├── tuned-bert              # tuned bert for creating embedding
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   └── visualize.py            # visualizer
├── log                         # tensorboard output files
│   └── $model_name$            # tensorboard output directory
│       └── events.out.tfevents.$time$.$server_name$
├── main.py                     # link prediction scripts
├── model                       # model files
│   └── *.pth
├── module                      # model implementation
│   ├── graph_encoder.py        # graph encoders & load function
│   ├── __init__.py
│   ├── module_with_device.py   # utils class: model with device management
│   └── relation_decoder.py     # relation decoder & load function
├── output                      # nohup output files
│   └── *.out
├── README.md
└── utils.py                    # link prediction utils & model utils...
```

## Usage
- 首先请在当前目录解压`data.zip`，然后创建`log`，`model`文件夹用于存储日志和模型。
- 目前仅支持训练和测试，没有部署和在线推断的部分
- 后续测试将尝试新的模型组合
```shell
$ python main.py --help

usage: Link prediction framework [-h] [-c CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to configuration file containing a dict of
                        parameters
```
### Config format
- 以下为`sample.json`，具体按照模型搭配，修改对应配置文件的各参数
```json
{
  "use_cuda": true,
  "data_path": "data",
  "model_path": "model/rgcn_dist_pre_raw.pth",
  "train": {
    "do": false,
    "log_file": "log/rgcn_dist_pre_raw",
    "n_epochs": 2000,
    "eval_every": 400,
    "edge_sampler": "uniform",
    "negative_sample": 10,
    "graph_split_size": 0.5,
    "train_batch_size": 25000,
    "grad_norm": 1.0,
    "lr": 1e-2,
    "weight_decay": 0.0,
    "regularization": 0.01
  },
  "test": {
    "do": true,
    "test_target": true
  },
  "load_embed": {
    "do": false,
    "embed_path": "embeds/emb_raw.npz"
  },
  "export_embed": {
    "do": false,
    "embed_path": "embeds/emb_rgcn_pre_raw.npz"
  },
  "graph_encoder": {
    "name": "rgcn",
    "device": "cuda:0",
    "n_hidden": 400,
    "details": {
      "rgcn": {
        "n_layers": 2,
        "n_bases": 100,
        "dropout": 0.2
      }
    }
  },
  "relation_decoder": {
    "name": "distmult",
    "device": "cuda:0",
    "n_hidden": 400,
    "details": {}
  }
}
```
