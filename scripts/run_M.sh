export CUDA_VISIBLE_DEVICES=1


for model in FEDformer Autoformer
do

for preLen in 96 192 336 720
do

# ETT m1
python -u train_models.py \
  --is_training 1 \
  --root_path /home/ilyakau/TS_curvature_analysis/dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --task_id ETTm1 \
  --model $model \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \

# ETTh1
python -u train_models.py \
  --is_training 1 \
  --root_path /home/ilyakau/TS_curvature_analysis/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1

# ETTm2
python -u train_models.py \
  --is_training 1 \
  --root_path /home/ilyakau/TS_curvature_analysis/dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1

# ETTh2
python -u train_models.py \
  --is_training 1 \
  --root_path /home/ilyakau/TS_curvature_analysis/dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --task_id ETTh2 \
  --model $model \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1

## electricity
python -u train_models.py \
 --is_training 1 \
 --root_path /home/ilyakau/TS_curvature_analysis/dataset/electricity/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 1

# exchange
python -u train_models.py \
 --is_training 1 \
 --root_path /home/ilyakau/TS_curvature_analysis/dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 1

# traffic
python -u train_models.py \
 --is_training 1 \
 --root_path /home/ilyakau/TS_curvature_analysis/dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 3

# weather
python -u train_models.py \
 --is_training 1 \
 --root_path /home/ilyakau/TS_curvature_analysis/dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1
done

done

