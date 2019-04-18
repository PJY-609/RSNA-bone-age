from datetime import datetime
from itertools import cycle
from keras.preprocessing.image import ImageDataGenerator
from keras import Input
from keras.applications import InceptionV3
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D, concatenate
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, CSVLogger, RemoteMonitor,ReduceLROnPlateau
from common import get_boneage_dataframe, flow_from_dataframe

# hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_VAL = 2
base_dir = 'E:/RSNA_bone_age'
IMG_SIZE = (299, 299)

print('==================================================')
print('============ Preprocessing Image Data ============')
print('==================================================')

print('current time: %s' % str(datetime.now()))

core_idg = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                              zoom_range=0.2, horizontal_flip=True)

valid_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

print('==================================================')
print('============ Creating Data Generators ============')
print('==================================================')

print('current time: %s' % str(datetime.now()))

print('==================================================')
print('========== Reading RSNA Boneage Dataset ==========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

boneage_df = get_boneage_dataframe('boneage-training-dataset', 'boneage-training-dataset.csv', 'id')

boneage_df = boneage_df[0:8]

train_df_boneage, valid_df_boneage = train_test_split(boneage_df, test_size=0.2, random_state=2019)

train_gen_boneage = flow_from_dataframe(core_idg, train_df_boneage, path_col='path', y_col='boneage',
                                        target_size=IMG_SIZE,
                                        color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)

valid_gen_boneage = flow_from_dataframe(valid_idg, valid_df_boneage, path_col='path', y_col='boneage',
                                        target_size=IMG_SIZE,
                                        color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)

print('==================================================')
print('================= Building Model =================')
print('==================================================')

print('current time: %s' % str(datetime.now()))

i1 = Input(shape=(299, 299, 3), name='input_img')
i2 = Input(shape=(1,), name='input_gender')
base = InceptionV3(input_tensor=i1, input_shape=(299, 299, 3), include_top=False, weights=None)
feature_img = base.get_layer(name='mixed10').output
feature_img = AveragePooling2D((2, 2))(feature_img)
feature_img = Flatten()(feature_img)
feature_gender = Dense(32, activation='relu')(i2)
feature = concatenate([feature_img, feature_gender], axis=1)

o = Dense(1000, activation='relu')(feature)
o = Dense(1000, activation='relu')(o)
o = Dense(1)(o)
model = Model(inputs=[i1, i2], outputs=o)
optimizer = Adam(lr=1e-3)
model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae'])

print('==================================================')
print('======= Training Model on Boneage Dataset ========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

model.summary()

early = EarlyStopping(monitor="val_loss", mode="min",
                      patience=10)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1,
                                   mode='auto', epsilon=0.0001, cooldown=5, min_lr=LEARNING_RATE * 0.1)


def combined_generators(image_generator, gender_data, batch_size):
    gender_generator = cycle(batch(gender_data, batch_size))
    while True:
        nextImage = next(image_generator)
        nextGender = next(gender_generator)
        assert len(nextImage[0]) == len(nextGender)
        yield [nextImage[0], nextGender], nextImage[1]


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


train_gen_wrapper = combined_generators(train_gen_boneage, train_df_boneage['male'], BATCH_SIZE_TRAIN)
val_gen_wrapper = combined_generators(valid_gen_boneage, valid_df_boneage['male'], BATCH_SIZE_VAL)

history = model.fit_generator(train_gen_wrapper, validation_data=val_gen_wrapper,
                              epochs=NUM_EPOCHS, steps_per_epoch=len(train_gen_boneage),
                              validation_steps=len(valid_gen_boneage),
                              callbacks=[early, reduceLROnPlat])
# print('Boneage dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])
#
# print('==================================================')
# print('================ Evaluating Model ================')
# print('==================================================')
#
# tend = datetime.now()
# print('current time: %s' % str(datetime.now()))
