# IR_Materials
All the assignments and projects on Information Retrieval 


## All data is stored at dataDump folders in their parent folders

write below command to execute different task with different datasets
```python
python likelihood.py --dataset hotpot --task queryLikelihood
python likelihood.py --dataset wikinq --task queryLikelihood
python likelihood.py --dataset hotpot --task docLikelihood
python likelihood.py --dataset wikinq --task docLikelihood

python clsLikelihood.py --dataset [hotpot/wikinq] --task [queryLikelihood/docLikelihood] --ns [inBatch/random]