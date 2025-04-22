MODEL="facebook/wav2vec2-base"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="wav2vec2_bs4_lr5e-5_ep50_seed${s}"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset autumn \
                                        --seed $s \
                                        --data_folder ../Data/ \
                                        --train_annotation_file train.csv \
                                        --test_annotation_file test.csv \
                                        --n_cls 2 \
                                        --epochs 50 \
                                        --batch_size 4 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --sample_rate 16000 \
                                        --model $m \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
