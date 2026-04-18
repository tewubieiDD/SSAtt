#!/bin/bash

[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=eegnet,shconvnet