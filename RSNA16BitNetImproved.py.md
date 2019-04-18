## RSNA16BitNetImproved.py

***

####  To get dataframe

* To read from csv 

  ```python
  pd.read_csv()
  ```

* Combine directory and load

  ```python
  os.path.join()
  ```

* Create a new attribute column mapping from an existing column of dataframe

  ```python
  dataframe[attr_name].map()
  ```

* Create path in accordance to the attribute 'id' of dataframe

  ```python
  dataframe[attr_name].map(lambda x: os.path.join(...,...,'{}.png'.format(x)))
  ```

* To create variable in string

  ```python
  '...{}...'.format(x)
  ```

* If the path exists

  ```python
  os.path.exists
  ```

* Type of the dataframe and its value

  dataframe[attr_name] belongs to pandas

  dataframe[attr_name].value belongs to ndarray 

***

#### To preprocess for data augmentation

* keras.preprocessing.image.ImageDataGenerator

  ```python
  ImageDataGenerator(rotation_range= ,width_shift_range= ,height_shift_range= ,
                     zoom_range= ,horizontal_flip= )
  ```

  ***

#### To split training set and testing set from the raw dataframe

* ```python
  sklearn.train_test_split(src_dataframe, test_szie= , random_state= )
  ```

  ***

####  Creates a DirectoryIterator from in_df at path_col with image preprocessing defined by img_data_gen, and labels are specified by y_col

* ```python
   def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args)
  ```

* To get base directory

  ```python
  os.path.dirname()
  ```

* To generate a DirectoryIterator 

  ```python
  img_data_gen.flow_from_directory(base_dir,...)
  ```

* To get the value of the dataframe columns

  ```python
  dataframe['...'].value[i]
  ```

* To create attribute of DirectoryIterator

  ```pyt
  df_gen.attr = ...
  ```

* To get the dimension of dataframe

  ```python
  dataframe.shape[i]
  ```

* In order to work for ReduceLROnPlateau and EarlyStopping

  ```
  df_gen.n = in_df.shape[0]
  ```

  ***

#### Multiple Inputs

* ```python
  concatenate([a, b], axis=1)
  ```

* ```python
  Model(inputs=[i1, i2], outputs= )
  ```

  ***

#### Fulfill multiple inputs with .fit_generator()

* ```python
  def batch(iterable, n=1):
      l = len(iterable)
      for ndx in range(0, l, n):
          yield iterable[ndx:min(ndx + n, l)]
  ```

* Convert the non-iterable type to iterable 

  ```python
  def combined_generator(iterable, batch_size):
  	generator_1 = cycle(batch(iterable, batch_size))
  	while Ture:
      	a = next(generator_1)
      	b = next(generator_2)
      	yield [a[0], b], a[1]
  ```

  ***

#### Use .fit_generator()

* ```python
  train_gen_wrapper = combined_generators(..., ..., ...)
  ```

* ```python
  model.fit_generator(train_gen_wrapper,...)
  ```

  