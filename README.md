### Dataset

[A Dataset for Document Grounded Conversations](https://arxiv.org/pdf/1809.07358.pdf). 

### Requirements
```shell
pip install -r requirements.txt
```

### How to run

1. Preprocess

   ```shell
   python preprocess.py \
   		--train_src data/src-train-tokenized.txt \
   		--valid_src data/src-valid-tokenized.txt \
   		--train_knl data/knl-train-tokenized.txt \
   		--valid_knl data/knl-valid-tokenized.txt \
   		--train_tgt data/tgt-train-tokenized.txt \
   		--valid_tgt data/tgt-valid-tokenized.txt \
   		--save_data data/cmu_movie \
   		-dynamic_dict \
   		-share_vocab \
   		-src_seq_length_trunc 50 \
   		-tgt_seq_length_trunc 50 \
   		-knl_seq_length_trunc 200 \
   		-src_seq_length 150 \
   		-knl_seq_length 800 
   ```

2. Train

   ```shell
   python train.py -config config/config-transformer-base-1GPU.yml
   ```

3. Generate

   ```shell
   python translate.py \
   		--src data/src-test-tokenized.txt \ 
   		--tgt data/tgt-test-tokenized.txt \
   		--knl data/knl-test-tokenized.txt \
   		--model models/base_model_step_20000.pt \
   		--output pred.txt \
   		-replace_unk \
   		-report_bleu \
   		-dynamic_dict \
   		-gpu 1 \
   		-batch_size 32
   ```

We use OpenNMT-py [OpenNMT-py](<http://opennmt.net/OpenNMT-py/>).
