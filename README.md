# crazycry

## datasets

I can't put it here.

## train script

`python main.py --epochs=50 --lr=0.01 --gpu=0 --lr-steps=30,45 -b 32 -p 1`

## test script

`python main.py --epochs=50 --lr=0.01 --gpu=0 --lr-steps=30,45 -b 32 -p 1 -e --resume ./weights/xxx.pth`