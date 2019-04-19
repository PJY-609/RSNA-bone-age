## RSNABaseline.py

***

#### Use zscore as labels

* How to caculate zscore

  ```py
  mean = dataframe['attr_name'].mean()
  div = dataframe['attr_name'].std()
  dataframe['zscore'] = dataframe['attr_name'].map(lambda x: (x - mean)/div)
  ```

* Categorize the attribute of dataframe

  ```python
  pd.cut(dataframe['attr_name'], interval)
  ```

* Get training set and testing set based on a balanced distribution of labels

  ```python
  train_test_split(..., stratify=dataframe['label_category'])
  ```

* Regroup training set by cluster the data with the same category

  ```python
  dataframe.groupby(['category1', 'category2']).apply(lambda x:x.sample(n, replace=True)).reset_index(drop=True)
  # replace = True necessary when the number of a certain satisfied category is less than n.
  # drop = True drop the original index
  # groupby operation use category1, and category2 as index
  ```

  ***

#### Apply Attention Model

```mermaid
graph TB
	st(Start)-->op1(VGG Pretrained Model)
	op1-->op2(Batch Normalization)
	op2-->op3(Conv2D_64)
	op3-->op4(COnv2D_16)
	op4-->op5(LocallyConnected2D_1)
	op2-->sub((Multiply))
	op1-.get_output_shape_0=pt_depth.->op6(Conv2D_pt_depth)
	op5-->op6
	op6-->sub
	sub-->op7(GlobalAveragePooling2D)
	op6-->op8(GlobalAveragePooling2D)
	op7--feature-->rescale((feature/mask))
	op8--mask-->rescale
	rescale-->op9(Dropout_0.5)
	op9-->op10(Dropout_0.25+Dense_1024)
	op10-->op11(Dense_1)
```

***

#### How to operate a certain layer

* To get the size of a certain layer

  ```python
  model.get_output_shape(i)[j]
  ```

* To get the output of a certain layer

  ```pyt
  model.get_layer(name='layer_name').output
  ```

  ***

#### Various ways to combine different layers

* To multiply outputs of 2 layers

  ```python
  multiply([layer1, layer2])
  ```

* To rescale of outputs of 2 layers

  ```py
  Lambda(lambda x: x[0]/x[1], name='name')([denominator, nominator])
  ```

  ***

#### To modify the metrics of a model using zscore as labels

* ```python
  model.compile(optimizer='', loss='', metrics=[mae])
  
  def mae(in_gt, in_pred):
      return mae(div*in_gt, div*in_pred)
  ```

  

