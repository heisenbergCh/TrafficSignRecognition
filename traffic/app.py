#coding=UTF-8 
from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
import os
import uuid
from PIL import Image, ExifTags
import perdict
import torch.nn as nn
from flask_cors import CORS
import json
## 继承nn.Module的Net类
class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''
    def __init__(self):
        super(LeNet, self).__init__()  # 这一个是python中的调用父类LeNet的方法，因为LeNet继承了nn.Module，如果不加这一句，无法使用导入的torch.nn中的方法，这涉及到python的类继承问题，你暂时不用深究

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)：输入层图片的输入尺寸，我看了那个文档，发现不需要天，会自动适配维度
            nn.Conv2d(3,6,5),   # padding=2保证输入输出尺寸相同：采用的是两个像素点进行填充，用尺寸为5的卷积核，保证了输入和输出尺寸的相同
            nn.ReLU(),                  # input_size=(6*28*28)：同上，其中的6是卷积后得到的通道个数，或者叫特征个数，进行ReLu激活
            nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)：经过池化层后的输出
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # input_size=(6*14*14)：  经过上一层池化层后的输出,作为第二层卷积层的输入，不采用填充方式进行卷积
            nn.ReLU(),            # input_size=(16*10*10)： 对卷积神经网络的输出进行ReLu激活
            nn.MaxPool2d(2, 2)    # output_size=(16*5*5)：  池化层后的输出结果
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 58)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(-1, 16 * 53 * 53)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


app = Flask(__name__)  # 实例Flask应用
CORS(app, supports_credentials=True)
# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']
# 设置图片保存文件夹
app.config['UPLOAD_FOLDER'] = './image/'
# 设置图片返回的域名前缀
# image_url = "image/"
# 设置图片压缩尺寸
image_c = 1000
# 跨域支持
# def after_request(resp):
#     resp.headers['Access-Control-Allow-Origin'] = '*'
#     return resp
# app.after_request(after_request)
# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS
# 首页
@app.route('/')
def hello_world():
    return render_template('index.html')
# 心跳检测
@app.route("/check", methods=["GET"])
def check():
    return 'Im live'


# 图片获取地址 用于存放静态文件
@app.route("/image/<imageId>")
def get_frame(imageId):

    # image_url = request.args.get('image_url')          #获取get提交过来的数据。
 
    #图片上传保存的路径
    try:
        with open(r'image/{}'.format(imageId), 'rb') as f:
            image = f.read()
            result = Response(image, mimetype="image/jpg")
            return result   
    except BaseException as e:
        return json.dumps({"code": '503', "data": str(e), "message": "图片不存在"})




# 图片获取地址 用于存放静态文件
@app.route("/image_predict", methods=["GET"])
def predict():

    image_url = request.args.get('image_url')          #获取get提交过来的数据。
    image_path='image/{}'.format(image_url)
    classe = perdict.perdict(image_path)
        
    return json.dumps({"code": '200', "classe": classe,"message": "识别完成"})


# 上传图片
@app.route("/upload_image", methods=['POST', "GET"])
def uploads():
    if request.method == 'POST':
        # 获取文件
        file = request.files['file']
        # 检测文件格式
        if file and allowed_file(file.filename):
            # secure_filename方法会去掉文件名中的中文，获取文件的后缀名
            file_name_hz = secure_filename(file.filename).split('.')[-1]
            # 使用uuid生成唯一图片名
            first_name = str(uuid.uuid4())
            # 将 uuid和后缀拼接为 完整的文件名
            file_name = first_name + '.' + file_name_hz
            # 保存原图
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            # 以下是压缩图片的过程，在原图的基础上
            file_min = Image.open(file)
            # # exif读取原始方位信息 防止图片压缩后发生旋转
            # try:
            #     for orientation in ExifTags.TAGS.keys():
            #         if ExifTags.TAGS[orientation] == 'Orientation': break
            #     exif = dict(file_min._getexif().items())
            #     if exif[orientation] == 3:
            #         file_min = file_min.rotate(180, expand=True)
            #     elif exif[orientation] == 6:
            #         file_min = file_min.rotate(270, expand=True)
            #     elif exif[orientation] == 8:
            #         file_min = file_min.rotate(90, expand=True)
            # except:
            #     pass
            # # 获取原图尺寸
            # w, h = file_min.size
            # # 计算压缩比
            # bili = int(w / image_c)
            # # 按比例对宽高压缩
            # file_min.thumbnail((w // bili, h // bili))
            file_min=file_min.resize((224,224),Image.LANCZOS)

            # 生成缩略图的完整文件名
            file_name_min = first_name + '_min.' + file_name_hz
            # 保存缩略图
            file_min.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name_min))
            # 返回原本和缩略图的 完整浏览链接
            return  json.dumps({"code": '200', "image_url": file_name, "image_url_min":file_name_min,
                    "message": "上传成功"})
        else:
            return "格式错误，仅支持jpg、png、jpeg格式文件"
    return json.dumps({"code": '503', "data": "", "message": "仅支持post方法"})  
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)  # 项目入口
