import numpy as np
import pandas as pd
import os

base_datasets_dir = 'E:/RSNA_bone_age/'

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    """
    Creates a DirectoryIterator from in_df at path_col with image preprocessing defined by img_data_gen. The labels
    are specified by y_col.
    :param img_data_gen: an ImageDataGenerator
    :param in_df: a DataFrame with images
    :param path_col: name of column in in_df for path
    :param y_col: name of column in in_df for y values/labels
    :param dflow_args: additional arguments to flow_from_directory
    :return: df_gen (keras.preprocessing.image.DirectoryIterator)
    """
    print('flow_from_dataframe() -->')
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    # flow_from_directory: Takes the path to a directory, and generates batches of augmented/normalized data.
    # sparse: a 1D integer label array is returned
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', **dflow_args)
    # df_gen: A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images
    # with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
    df_gen.filenames = in_df[path_col].values
    print(type(in_df[y_col].values))
    df_gen.classes = np.stack(in_df[y_col].values)
    # df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    # df_gen._set_index_array()
    df_gen.directory = base_dir  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    print('flow_from_dataframe() <--')
    return df_gen


def get_boneage_dataframe(dir_name, csv_name, img_col_name):
    base_boneage_dir = base_datasets_dir
    boneage_df = pd.read_csv(os.path.join(base_boneage_dir, csv_name))
    boneage_df['path'] = boneage_df[img_col_name].map(lambda x: os.path.join(base_boneage_dir, dir_name,
                                                                     '{}.png'.format(x)))  # create path from id
    boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
    print(dir_name, boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')
    # boneage_df['boneage_category'] = pd.cut(boneage_df[class_str_col], 10)
    return boneage_df

def get_age_dataframe(dir_name, csv_name):
    age_df = pd.read_csv(os.path.join(base_datasets_dir, csv_name))

    age_df['path'] = age_df['id'].map(lambda x: os.path.join(base_datasets_dir, dir_name,
                                                             '{}.png'.format(x)))  # add path to dictionary
    age_df['exists'] = age_df['path'].map(os.path.exists)  # add exists to dictionary
    print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')  # print how many images have been found
    age_df['gender'] = age_df['male'].map(
        lambda x: 'male' if x else 'female')  # convert boolean to string male or female

    boneage_mean = age_df['boneage'].mean()
    boneage_div = 2 * age_df['boneage'].std()
    age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)  # creates classes
    age_df.dropna(inplace=True)  # inplace = True 直接对原对象进行修改 删去空缺属性的元素
    age_df.sample(3)
    age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)

    return [age_df, boneage_div]
