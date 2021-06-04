for seed in {1347..1356}
do
  python bert.py ../data/train.csv bert --criterion auc_macro --seed $seed --run_test
done
