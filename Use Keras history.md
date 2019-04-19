## Use Keras history

* How to write history into a file

  ```python
  history = model.fit_generator(...)
  
  with open('E:/RSNA_bone_age/history.txt','wb') as file_pi:
      pickle.dump(history.history, file_pi)
  ```

* How to load history 

  ```python
  with open('E:/RSNA_bone_age/history',
            'rb') as file_pi:
      f = pickle.load(file_pi)
      
  for val in f:
      print("Print {}:".format(val), f[val])
  ```

* history have attribute 

  * val_loss
  * val_mean_absolute_error
  * loss
  * mean_absolute_error
  * lr

  ***

  ### More on with...as...

  ```python
  class Sample:
      def __enter__(self):
          print("In __enter__()")
          return "Foo"
  
      def __exit__(self, type, value, trace):
          print("In __exit__()")
  
  
  def get_sample():
      return Sample()
  
  
  with get_sample() as sample:
      print("sample:", sample)
  ```

  *result*

  ```python
  In __enter__()
  sample: Foo
  In __exit__()
  ```

  * with...as... is a simplified version of error report syntax

  ```python
  class Sample:
      def __enter__(self):
          return self
  
      def __exit__(self, type, value, trace):
          print("type:", type)
          print("value:", value)
          print("trace:", trace)
  
      def do_something(self):
          bar = 1 / 0
          return bar + 10
  
  
  with Sample() as sample:
      sample.do_something()
  ```

  *result*

  ```python
  ZeroDivisionError: division by zero
  type: <class 'ZeroDivisionError'>
  value: division by zero
  trace: <traceback object at 0x000001B028DDBD08>
  ```

  ***

  #### More on pickle.dump & load

  * a tool to write and load file