for seed in {6360..6364}
do
  python bert.py ../data/train.csv bert --criterion auc_macro --local_weights results/bert_ttp_2m/model.pth --seed $seed --run_test
done
