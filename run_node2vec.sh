ratios=(0.2 0.4 0.6 0.8)
for r in ${ratios[@]}
do
	python run_node2vec_fast.py --dataset $1_$r --dimensions 128 --num-walks 10 --seed 42 --walk-length 40 --window-size 10 --workers 16 --weighted
done
