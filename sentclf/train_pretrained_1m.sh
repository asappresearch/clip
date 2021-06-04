for seed in {1337..1346}
do
  python bert.py ../data/train.csv bert --criterion auc_macro --local_weights results/bert_ttp_1m/model.pth --seed $seed --run_test
done
