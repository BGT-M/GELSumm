lrs=(0.1 0.05 0.02 0.01)
for lr in ${lrs[@]};
do
	python main.py --dataset reddit --lr $lr --dropout 0.2 --hidden 64
done

