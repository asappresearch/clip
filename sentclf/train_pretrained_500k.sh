for seed in {1347..1356}
do
  python bert.py ../data/train.csv bert --criterion auc_macro --local_weights results/bert_ttp_500k/model.pth --seed $seed --run_test
done
