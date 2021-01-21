ratios=(0.2 0.4 0.6 0.8)
lrs=(1 0.5 0.2 0.1)
epochs=(1 2 5 10)
for r in ${ratios[@]}
do
    python run_deepwalk.py --dataset $1_$r --seed $2 --power 2
done
