python retrain-inception.py --use-attn True --train-phase FFaaabaaabaa --attn-loss l2 --base-net inception_v3 --attn-shape full
python retrain-layer4.py --use-attn False --train-phase ffffffffff --base-net resnet50
python retrain-layer4.py --use-attn True --train-phase FFaaabaaabaa --base-net resnet50 --attn-loss l2 --attn-shape full
