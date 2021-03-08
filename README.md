# flipkart_Extract-a-Thon

## Problem setting
Our task is to extract attribute-value pair from given product description. We approached this problem as a Question-Answer task where the set of attributes are assumed to be Questions on given product description.

## Approach
We solved this problem as a transfer learning task to get task agnostic features and finetuned last layers of model to learn task specfic patterns. 

### Model: 
We followed [BERT-large-uncased model](https://huggingface.co/bert-large-uncased) architecture. 
### Training procedure:  
To solve the unsupervised attribute value selection problem, we trained BERT in self-supervised fashion for 3 stages. We used adaptive threshold based algorithm to get pseudo labels.

## How to evaluate 
1. Clone the repo from [git-repo](https://github.com/1201amit/flipkart_Extract-a-Thon/new/master). 
2. Create a folder "./checkpoints" and download model weights from [link](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/prajjwalm_iisc_ac_in/EZpn_1h28o1Dtyoz5VsT5JgBLMQP5ksqbnjgvpPUMrnB4Q?e=AmHA5A). 
3. Download Flipkart_complete_data. 
4. Run following command to evaluate
```python
python eval.py
```
