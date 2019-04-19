from common import get_age_dataframe, flow_from_dataframe
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, \
    Lambda
from keras.layers import BatchNormalization
from keras.models import Model
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import numpy as np

age_df, boneage_div = get_age_dataframe(dir_name='boneage-training-dataset', csv_name='boneage-training-dataset.csv')

age_df = age_df[0:8]

raw_train_df, valid_df = train_test_split(age_df, test_size=0.2, random_state=2018, stratify=age_df['boneage_category'])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])


train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace=True)
                                                                    ).reset_index(drop=True)

print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
# groupby 对 dataframe先按两个类分组，再进行取样操作，
# sample replace=True 当实际样本数小于取样数进行有放回抽样
# 对 groupby 以后的dataframe 进行索引重排序 drop = True 丢掉旧的索引


IMG_SIZE = (384, 384)  # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False,
                              samplewise_std_normalization=False,
                              horizontal_flip=True,
                              vertical_flip=False,
                              height_shift_range=0.15,
                              width_shift_range=0.15,
                              rotation_range=5,
                              shear_range=0.01,
                              fill_mode='nearest',
                              zoom_range=0.25,
                              preprocessing_function=preprocess_input)  # VGG16 的预处理函数


train_gen = flow_from_dataframe(core_idg, train_df, path_col='path', y_col='boneage_zscore', target_size=IMG_SIZE,
                                color_mode='rgb', batch_size=32)
valid_gen = flow_from_dataframe(core_idg, valid_df, path_col='path', y_col='boneage_zscore', target_size=IMG_SIZE,
                                color_mode='rgb', batch_size=256)

test_X, test_Y = next(valid_gen)

t_x, t_y = next(train_gen)
in_lay = Input(t_x.shape[1:])
base_pretrained_model = VGG16(input_shape=t_x.shape[1:], include_top=False, weights='imagenet')
base_pretrained_model.trainable = False
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)

bn_features = BatchNormalization()(pt_features)

attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
attn_layer = LocallyConnected2D(1,
                                kernel_size=(1, 1),
                                padding='valid',
                                activation='sigmoid')(attn_layer)

up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
               activation='linear', use_bias=False, weights=[up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.25)(Dense(1024, activation='elu')(gap_dr))
out_layer = Dense(1, activation='linear')(dr_steps)  # linear is what 16bit did
bone_age_model = Model(inputs=[in_lay], outputs=[out_layer])

def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div * in_gt, boneage_div * in_pred)


bone_age_model.compile(optimizer='adam', loss='mse',
                       metrics=[mae_months])

bone_age_model.summary()

# weight_path = base_bone_dir + "{}_weights.best.hdf5".format('bone_age')

# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
#                              save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001,
                                   cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=5)  # probably needs to be more patient, but kaggle time is limited
callbacks_list = [early, reduceLROnPlat]

bone_age_model.fit_generator(train_gen,
                             validation_data=(test_X, test_Y),
                             epochs=15,
                             callbacks=callbacks_list)