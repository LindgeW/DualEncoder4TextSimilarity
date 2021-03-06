#python train.py --cuda 0 -lr 5e-4 --batch_size 16 --update_step 1 --dropout 0.2 --bert_embed_dim 128 --cmp_mode multineg --model_chkp model_tiny_multineg_128.pkl --vocab_chkp vocab_tiny_multineg.pkl &> tiny_1.log &

#python train.py --cuda 7 -lr 5e-4 --batch_size 16 --update_step 1 --dropout 0.2 --bert_embed_dim 256 --cmp_mode multineg --model_chkp model_tiny_multineg_256.pkl --vocab_chkp vocab_tiny_multineg.pkl &> tiny_2.log &

#python train.py --cuda 1 -lr 5e-4 --batch_size 16 --update_step 1 --dropout 0.2 --bert_embed_dim 128 --cmp_mode multineg --model_chkp model_small_multineg_128.pkl --vocab_chkp vocab_small_multineg.pkl &> small_1.log &

#python train.py --cuda 2 -lr 5e-4 --batch_size 16 --update_step 1 --dropout 0.2 --bert_embed_dim 256 --cmp_mode multineg --model_chkp model_small_multineg_256.pkl --vocab_chkp vocab_small_multineg.pkl &> small_2.log &

#python train.py --cuda 3 -lr 5e-4 --batch_size 16 --update_step 1 --dropout 0.2 --bert_embed_dim 512 --cmp_mode multineg --model_chkp model_small_multineg_512.pkl --vocab_chkp vocab_small_multineg.pkl &> small_3.log &

python train.py --cuda 4 -lr 5e-4 --batch_size 8 --update_step 2 --dropout 0.3 --bert_embed_dim 128 --cmp_mode multineg --model_chkp model_wwm_multineg_128.pkl --vocab_chkp vocab_wwm_multineg.pkl &> wwm_1.log &

python train.py --cuda 5 -lr 5e-4 --batch_size 8 --update_step 2 --dropout 0.3 --bert_embed_dim 256 --cmp_mode multineg --model_chkp model_wwm_multineg_256.pkl --vocab_chkp vocab_wwm_multineg.pkl &> wwm_2.log &

python train.py --cuda 6 -lr 5e-4 --batch_size 8 --update_step 2 --dropout 0.3 --bert_embed_dim 512 --cmp_mode multineg --model_chkp model_wwm_multineg_512.pkl --vocab_chkp vocab_wwm_multineg.pkl &> wwm_3.log &

#python train.py --cuda 1 -lr 5e-4 --batch_size 8 --update_step 2 --dropout 0.3 --bert_embed_dim 128 --cmp_mode multineg --model_chkp model_128.pkl --vocab_chkp vocab_128.pkl &> 10.log &

#python train.py --cuda 2 -lr 5e-4 --batch_size 8 --update_step 2 --dropout 0.3 --bert_embed_dim 256 --cmp_mode multineg --model_chkp model_256.pkl --vocab_chkp vocab_256.pkl &> 11.log &

#python train.py --cuda 3 -lr 5e-4 --batch_size 8 --update_step 2 --dropout 0.3 --bert_embed_dim 512 --cmp_mode multineg --model_chkp model_512.pkl --vocab_chkp vocab_512.pkl &> 12.log &