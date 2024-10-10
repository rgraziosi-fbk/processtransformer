## ProcessTransformer: Predictive Business Process Monitoring with Transformer Network

![header image](https://github.com/Zaharah/processtransformer/blob/main/pt.JPG)

<details><summary>Abstract (click to expand)</summary>
<p>

Predictive business process monitoring focuses on predicting future characteristics of a running process using event logs. The foresight into process execution promises great potentials for efficient operations, better resource management, and effective customer services. Deep learning-based approaches have been widely adopted in process mining to address the limitations of classical algorithms for solving multiple problems, especially the next event and remaining-time prediction tasks. Nevertheless, designing a deep neural architecture that performs competitively across various tasks is challenging as existing methods fail to capture long-range dependencies in the input sequences and perform poorly for lengthy process traces. In this paper, we propose ProcessTransformer, an approach for learning high-level representations from event logs with an attention-based network. Our model incorporates long-range memory and relies on a self-attention mechanism to establish dependencies between a multitude of event sequences and corresponding outputs. We evaluate the applicability of our technique on nine real event logs. We demonstrate that the transformer-based model outperforms several baselines of prior techniques by obtaining on average above 80% accuracy for the task of predicting the next activity. Our method also perform competitively, compared to baselines, for the tasks of predicting event time and remaining time of a running case.

</p>
</details>


#### Tasks
- Next Activity Prediction
- Time Prediction of Next Activity
- Remaining Time Prediction

### Install 
```
pip install processtransformer
```

### Usage for trace generation

1. In the `__main__` part of `data_processing.py`, verify that column_names and csv_separator are suited for the current dataset
2. Preprocess data for next_activity task: `python data_processing.py --dataset=sepsis --dir_path=./datasets/next_activity --raw_log_file=./datasets/sepsis/sepsis.csv --task=next_activity --insert_eot --sort_temporally`. You'll find processed data in ./datasets/next_activity/{dataset-name}/processed
3. Preprocess data for next_time task: `python data_processing.py --dataset=sepsis --dir_path=./datasets/next_time --raw_log_file=./datasets/sepsis/sepsis.csv --task=next_time --insert_eot --sort_temporally`. You'll find processed data in ./datasets/next_time/{dataset-name}/processed
4. In `next_activity.py` and `next_time.py`, verify that parameters like validation_split, early stopping patience and min_delta suit your needs
5. Train the next_activity model: `python next_activity.py --dataset_dir=./datasets/next_activity --dataset=sepsis --epochs=100`. You'll find trained model in ./models/{dataset-name}/next_activity_*
6. Train the next_time model: `python next_time.py --dataset_dir=./datasets/next_time --dataset=sepsis --epochs=100`. You'll find trained model in ./models/{dataset-name}/next_time_*
7. Ensure constants, DATASET_NAME, NUM_TRACES and START_TIMESTAMP are correctly configured in `generate_log.py`
8. Generate logs: `python generate_log.py`

### Are train/val/test splits temporal or random?

They are temporal because:

- `data_processing.py` accepts an argument `--sort-temporally` that splits between train and test using temporal order
- `next_activity.py` and `next_time.py` make use of TensorFlow's Model.fit method, with `validation_split` set to 0.25: the validation split, as [per documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit), is selected from the last x and y samples provided, and is performed before shuffling data.


### Usage  
We provide the necessary code to use ProcessTransformer with the event logs of your choice. We illustrate the examples using the helpdesk dataset. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tiOh2VS8yzOVON26CbmWn0oUn-dWAFhN?usp=sharing)

For the data preprocessing,  run:

```python
python data_processing.py --dataset=sepsis --dir_path=./datasets/next_activity --raw_log_file=./datasets/sepsis/sepsis.csv --task=next_activity --insert_eot --sort_temporally
python data_processing.py --dataset=helpdesk --dir_path=./datasets/next_time --task=next_time --insert_eot
python data_processing.py --dataset=helpdesk --task=remaining_time
```
To train and evaluate the model, run:

```python
python next_activity.py --dataset=helpdesk --epochs=100
python next_time.py --dataset_dir=./datasets/next_time --dataset=helpdesk --epochs=100
python remaining_time.py --dataset=helpdesk --epochs=100
```


### Tools
- <a href="http://tensorflow.org/">Tensorflow >=2.4</a>

## Data 
The events log for the predictive busienss process monitoring can be found at [4TU Research Data](https://data.4tu.nl/categories/_/13500?categories=13503)

## How to cite 

Please consider citing our paper if you use code or ideas from this project:

Zaharah A. Bukhsh, Aaqib Saeed, & Remco M. Dijkman. (2021). ["ProcessTransformer: Predictive Business Process Monitoring with Transformer Network"](https://arxiv.org/abs/2104.00721). arXiv preprint arXiv:2104.00721 


```
@misc{bukhsh2021processtransformer,
      title={ProcessTransformer: Predictive Business Process Monitoring with Transformer Network}, 
      author={Zaharah A. Bukhsh and Aaqib Saeed and Remco M. Dijkman},
      year={2021},
      eprint={2104.00721},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
