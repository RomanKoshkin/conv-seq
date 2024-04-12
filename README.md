# _convSeq_: Fast and Scalable Method for Detecting Patterns in Spike Data

![](videos/output.gif)

**PAPER:** https://arxiv.org/abs/2402.01130

# Abstract

Spontaneous neural activity, crucial in memory, learning, and spatial navigation, often manifests itself as repetitive spatiotemporal patterns. Despite their importance, analyzing these patterns in large neural recordings remains challenging due to a lack of efficient and scalable detection methods. Addressing this gap, we introduce convSeq, an unsupervised method that employs backpropagation for optimizing spatiotemporal filters that effectively identify these neural patterns. Our method's performance is validated on various synthetic data and real neural recordings, revealing spike sequences with unprecedented scalability and efficiency. Significantly surpassing existing methods in speed, convSeq sets a new standard for analyzing spontaneous neural activity, potentially advancing our understanding of information processing in neural circuits.

## Installing dependencies

You don't have to install _all_ of the dependencies in `requirements.txt`. We just provide the package versions in case you get conflicts or other package-related issues.


## Dataset

Create a synthetic dataset. You can change the sequence parameters: number of neurons in a sequence (`seqlen`), spike dropout (`p_drop`), inter-sequence interval in timesteps (`gap_ts`) and spike timing jitter (`jitter_std`).

```bash
cd demo
python make_dataset.py \
    --seqlen 120 \
    --p_drop 0.1 \
    --gap_ts 400 \
    --jitter_std 10
```

## Filter optimization

optimize the filters for a given number of epochs:

```bash
cd demo
python demo.py --epochs 4000
```

The config is in `configs/config_demo.yaml`. When the script terminates it saves the detection results before and after optimization in the `artifacts` folder:

![asdf](artifacts/Fig2_stats.png)


Also, you can make a video of the optimization progress:

```bash
cd scripts
sh make_vids.sh
```

the video will be in the `videos` folder. NOTE: Make sure you have [`ffmpeg`](https://ffmpeg.org/) and [`ffpb`](https://pypi.org/project/ffpb/) installed.
