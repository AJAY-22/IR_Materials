# IR_Materials
All the assignments and projects on Information Retrieval 


## All data should be stored at dataDump folders in the parent folders
## Assignment1
Files available at `Assignment1/scripts/`

write below command to execute different task (queryLikelihood or docLikelihood) with different datasets (hotpotQA or wikiNQ)

For negative sampling `--ns` argument can be take two values `inBatch` or `random`

```python
python likelihood.py --dataset hotpot --task queryLikelihood
python likelihood.py --dataset wikinq --task queryLikelihood
python likelihood.py --dataset hotpot --task docLikelihood
python likelihood.py --dataset wikinq --task docLikelihood

python clsLikelihood.py --dataset [hotpot/wikinq] --task [queryLikelihood/docLikelihood] --ns [inBatch/random]