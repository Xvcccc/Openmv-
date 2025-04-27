import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载并预处理MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 归一化到0-1范围并添加通道维度
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)  # (10000, 28, 28, 1)


# 2. 构建模型（轻量级适合OpenMV）
def create_model():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28, 1)),

        # 特征提取部分
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # 分类部分
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


model = create_model()
model.summary()

# 3. 编译和训练模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 添加早停和模型保存回调
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=20,
                    validation_split=0.1,
                    callbacks=callbacks)

# 4. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n测试准确率: {test_acc:.4f}")

# 5. 转换为TFLite格式（OpenMV兼容）
# 替换原来的转换代码
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # 禁用所有优化
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # 仅使用基础算子
tflite_model = converter.convert()

with open('mnist_openmv_compatible.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite模型已保存为: mnist_model.tflite")

# 6. 验证转换后的模型
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 测试一个样本
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
test_image = x_test[0:1]  # 取第一个测试样本
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"测试样本预测结果: {np.argmax(output_data)} (真实标签: {y_test[0]})")

# 7. 可视化训练过程
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
