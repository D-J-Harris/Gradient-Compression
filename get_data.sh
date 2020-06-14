echo "- Downloading Penn Treebank (PTB)"
mkdir -p penn
cd penn
wget --quiet --continue -O train.txt https://raw.githubusercontent.com/yangsaiyong/tf-adaptive-softmax-lstm-lm/master/ptb_data/ptb.train.txt
wget --quiet --continue -O valid.txt https://raw.githubusercontent.com/yangsaiyong/tf-adaptive-softmax-lstm-lm/master/ptb_data/ptb.valid.txt
wget --quiet --continue -O test.txt https://raw.githubusercontent.com/yangsaiyong/tf-adaptive-softmax-lstm-lm/master/ptb_data/ptb.test.txt
cd ..