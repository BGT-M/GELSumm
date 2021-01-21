ratios=(0.2 0.4 0.6 0.8)
for r in ${ratios[@]}
do
	python run_node2vec.py --dataset $1_$r --seed $2 --power 2
done
