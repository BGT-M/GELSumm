ratios=(0.2 0.4 0.6 0.8)
for r in ${ratios[@]}
do
	python run_node2vec.py --dataset $1_$r --dimensions 128 --num-walks 10 --seed $2 --walk-length 80 --window-size 10 --workers 16 --weighted --lr 0.1 --epochs 1
done
