
MODEL="efficientnet_b0"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="eff_bs128_lr1e-3_ep200_seed${s}_oaf"
        CUDA_VISIBLE_DEVICES=1 python main.py --tag $TAG \
                                        --dataset tess \
                                        --seed $s \
                                        --data_folder ./data/ \
                                        --class_split emo \
                                        --train_annotation_file OAF_train_data.csv \
                                        --test_annotation_file OAF_test_data.csv \
                                        --n_cls 7 \
                                        --epochs 200 \
                                        --batch_size 128 \
                                        --optimizer adam \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --learning_rate 1e-3 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --sample_rate 16000 \
                                        --model $m \
                                        --pad_types repeat \
                                        --from_sl_official \
                                        --audioset_pretrained \
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
