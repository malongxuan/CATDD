action=$1

if [ "$action" = "preprocess" ]; then
	python3 preprocess.py \
		--train_src ../data/src-train-tokenized.txt \
		--valid_src ../data/src-valid-tokenized.txt \
		--train_knl ../data/knl-train-tokenized-truncate.txt \
		--valid_knl ../data/knl-valid-tokenized-truncate.txt \
		--train_tgt ../data/tgt-train-tokenized.txt \
		--valid_tgt ../data/tgt-valid-tokenized.txt \
		--save_data ../data/cmu_movie \
		-dynamic_dict \
		-share_vocab \
		-src_seq_length_trunc 50 \
		-tgt_seq_length_trunc 50 \
		-knl_seq_length_trunc 200 \
		-src_seq_length 150 \
		-knl_seq_length 200 
elif [ "$action" = "train" ]; then
	CUDA_VISIBLE_DEVICES=1 python3 train.py -config config/config-transformer-base-1GPU.yml
elif [ "$action" = "generate" ]; then
	CUDA_VISIBLE_DEVICES=1 python3 translate.py \
			--model models_add-norm/base_model_step_12000.pt \
			--knl ../data/knl-test-tokenized-truncate.txt \
			--src ../data/src-test-tokenized.txt \
			--tgt ../data/tgt-test-tokenized.txt \
			--output pred-add_norm.txt \
			-replace_unk \
			-report_bleu \
			-dynamic_dict \
			-gpu 1 \
			-batch_size 32
fi
