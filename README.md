# Dual Pseudo-Labels Interactive Self-Training for Semi-Supervised Visible-Infrared Person Re-Identification ICCV2023

## Contribution
1. We propose a dual pseudo-label interactive self-training framework for semi-supervised visible-infrared person Re-ID, which leverages the intro- and inter-modality characteristics to obtain hybrid pseudo-labels.
2. We introduce three modules: noise label penalty (NLP), noise correspondence calibration (NCC), and unreliable anchor learning (UAL). These modules help to penalize noise labels, calibrate noisy correspondences, and exploit hard-to-discriminate features.
3. We provide comprehensive evaluations under these two semi-supervised VI-ReID. Extensive experiments on two popular VI-ReID benchmarks demonstrate that our DPIS achieves impressive performance.

## Framework
![DPIS](framework_DPIS.png)

## Train
1. sh run\_train\_sysu.sh for SYSU-MM01
2. sh run\_train\_regdb.sh for RegDB
# Test
1. sh run\_test\_sysu.sh for SYSU-MM01
2. sh run\_test\_regdb.sh for RegDB

## Contact
jiangming.shi@outlook.com; S_yinxb@163.com.

The code is implemented based on OTLA.
