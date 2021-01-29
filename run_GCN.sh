ratios=(0.2 0.4 0.6 0.8)
for r in ${ratios[@]}
do
    # python run_GCN.py --dataset $1_$r --cuda 0 --epochs 100 --lr 0.01 --hidden 128 --dropout 0.0 --type rw --power 4 --seed $2 --log_turn 10
    python run_GCN.py --dataset $1_$r --cuda 0 --epochs 100 --lr 0.01 --hidden 128 --dropout 0.3 --type symm --power 0 --seed $2 --log_turn 10
done
