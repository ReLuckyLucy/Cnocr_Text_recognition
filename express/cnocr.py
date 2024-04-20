import matplotlib.pyplot as plt
import cv2
import os
import paddlehub as hub

# 显示matplotlib图形
plt.show()

with open('test.txt', 'rb') as f:  # 以二进制模式打开文件
    test_img_path = []
    for line in f:
        test_img_path.append(line.decode('utf-8').strip())  # 指定文件编码为utf-8
print(test_img_path)

ocr = hub.Module(name="chinese_ocr_db_crnn_server")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取测试文件夹test.txt中的照片路径
np_images = [cv2.imread(image_path) for image_path in test_img_path]

results = ocr.recognize_text(
    images=np_images,
    use_gpu=True,
    output_dir='ocr_result',
    visualization=False,
    box_thresh=0.5,
    text_thresh=0.5)

# 创建结果保存目录
result_dir = './result/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for i, result in enumerate(results):
    data = result['data']
    with open(os.path.join(result_dir, str(i) + ".txt"), 'w', encoding='utf-8') as f:
        for information in data:
            f.write(information['text'] + "\n")
            print('text: ', information['text'], '\nconfidence: ', information['confidence'], '\ntext_box_position: ',
                  information['text_box_position'])
