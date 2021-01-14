ratios=(0.2 0.4 0.6 0.8)
for r in ${ratios[@]}
do
	python run_deepwalk.py --dataset $1_$r --format adjlist --number-walks 10 --representation-size 128 --seed 42 --walk-length 40 --window-size 10 --workers 16
done
