for seed in {1337..1346}
do
  python bert.py ../data/train.csv clinicalbert --criterion auc_macro --seed $seed --run_test
done
