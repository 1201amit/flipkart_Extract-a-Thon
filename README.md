# flipkart_Extract-a-Thon

## Approach
### Dataset used: 
1. [Ali-express data](https://raw.githubusercontent.com/lanmanok/ACL19_Scaling_Up_Open_Tagging/master/publish_data.txt). It is sports-based supervised data. We used it as a source domain. 
2. Flipkart complete data. It is unsupervised data provided by flipkart. We used it as a target domain.

Our overall objective is to minimize the domain gap in order to learn task specfic and domain generic features. 
### Model: 
We used BERT-large-uncased model pretrained on SQUAD dataset. 
### Training procedure:  
To solve the unsupervised attribute value selection problem, we partially trained a BERT model on source domain to learn task specfic features and later finetuned it on unsupervised target domain to learn domain generic features. This finetuning consists of 3 self-supervised stages. 

## How to evaluate 
1. Clone the repo from [git-repo](https://github.com/1201amit/flipkart_Extract-a-Thon/new/master). 
2. Create a folder "./checkpoints" and download model weights from [link](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/prajjwalm_iisc_ac_in/EZpn_1h28o1Dtyoz5VsT5JgBLMQP5ksqbnjgvpPUMrnB4Q?e=AmHA5A). 
3. Download Flipkart_complete_data. 
4. Run following command to evaluate
```python
python eval.py
```
