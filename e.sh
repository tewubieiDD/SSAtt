#!/bin/bash

[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=eegnet,shconvnet

python SSAtt_bci.py --sub=1 &
python SSAtt_bci.py --sub=2 &
python SSAtt_bci.py --sub=3 &
python SSAtt_bci.py --sub=4 &
python SSAtt_bci.py --sub=5 &
python SSAtt_bci.py --sub=6 &
python SSAtt_bci.py --sub=7 &
python SSAtt_bci.py --sub=8 &
python SSAtt_bci.py --sub=9
wait

python SSAtt_mamem.py --sub=1 &
python SSAtt_mamem.py --sub=2 &
python SSAtt_mamem.py --sub=3 &
python SSAtt_mamem.py --sub=4 &
python SSAtt_mamem.py --sub=5 &
python SSAtt_mamem.py --sub=6 &
python SSAtt_mamem.py --sub=7 &
python SSAtt_mamem.py --sub=8 &
python SSAtt_mamem.py --sub=9 &
python SSAtt_mamem.py --sub=10 &
python SSAtt_mamem.py --sub=11
wait

python SSAtt_bcicha.py --sub=2 &
python SSAtt_bcicha.py --sub=6 &
python SSAtt_bcicha.py --sub=7 &
python SSAtt_bcicha.py --sub=11 &
python SSAtt_bcicha.py --sub=12 &
python SSAtt_bcicha.py --sub=13 &
python SSAtt_bcicha.py --sub=14 &
python SSAtt_bcicha.py --sub=16 &
python SSAtt_bcicha.py --sub=17 &
python SSAtt_bcicha.py --sub=18 &
python SSAtt_bcicha.py --sub=20 &
python SSAtt_bcicha.py --sub=21 &
python SSAtt_bcicha.py --sub=22 &
python SSAtt_bcicha.py --sub=23 &
python SSAtt_bcicha.py --sub=24 &
python SSAtt_bcicha.py --sub=26 &
wait