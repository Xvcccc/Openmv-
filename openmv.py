import sensor, image, time, tf, uos, gc

# 初始化摄像头
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.set_windowing((28, 28))     # 必须与模型输入尺寸匹配
sensor.set_auto_gain(False)
sensor.set_auto_exposure(False)
sensor.set_auto_whitebal(False)
sensor.skip_frames(time=1000)

# 加载模型
try:
    net = tf.load("/mnist_openmv.tflite")
except Exception as e:
    raise Exception("模型加载失败: %s" % e)

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 图像预处理函数（与训练时一致）
def preprocess(img):
    # 1. 转换为灰度
    img = img.to_grayscale()
    
    # 2. 二值化处理（固定阈值）
    img = img.binary([(120, 255)], invert=True)
    
    # 3. 形态学去噪
    img = img.morph(1, [1, 1, 1, 
                        1, 1, 1,
                        1, 1, 1])
    return img

# 主循环
clock = time.clock()
while(True):
    clock.tick()
    img = sensor.snapshot()
    
    # 预处理
    processed = preprocess(img)
    
    # 运行推理
    try:
        for obj in net.classify(processed):
            predictions = obj.output()
            max_idx = predictions.index(max(predictions))
            confidence = predictions[max_idx]
            label = labels[max_idx]
            
            # 显示结果
            img.draw_string(0, 0, label, color=255, scale=3)
            print("识别结果: %s (置信度: %.1f%%)" % (label, confidence*100))
    
    except Exception as e:
        print("识别错误:", e)
    
    print("FPS:", clock.fps())
    gc.collect()
