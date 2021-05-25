for seed in 1337 1338 1339 1340 1341 1342 1343 1344 1345 1346
do
  python bert.py ../../../data/all_revised_data/train.csv  clinicalbert_disch --criterion auc_macro --seed $seed
done
