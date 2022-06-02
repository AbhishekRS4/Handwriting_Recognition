from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

def training(name,model):
    path_train_data = "char_recog/train"
    path_test_data = "char_recog/val"
    batch_size = 32
    input_size = (64, 64)
    gen_train_data = ImageDataGenerator(rescale=1./255, width_shift_range=0.1,
        shear_range=0.15, zoom_range=0.1,
        channel_shift_range=10., horizontal_flip=False)

    gen_test_data = ImageDataGenerator(rescale=1./255)
    train_generator = gen_train_data.flow_from_directory(
                path_train_data,
                target_size=input_size,
                batch_size=batch_size,
                class_mode='categorical')

    val_generator = gen_train_data.flow_from_directory(
                path_test_data,  # this is the target directory
                target_size=input_size,  # all images will be resized to 64x64
                batch_size=batch_size,
                class_mode='categorical',subset='training')

#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)

    weights_pretrain = model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.samples // batch_size,
        epochs=100,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size)
    model.save_weights(f"{name}.h5")
    return weights_train
