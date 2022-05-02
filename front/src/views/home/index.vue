<!-- home -->
<template>
  <div class="index-container">
    <div class="warpper">
      <br><br>
      <h1 class="demo-home__title"><img src="https://img-blog.csdnimg.cn/76a2bed0be4242c19bc77b4af5afab8f.png" /><span>交通标志分类识别</span></h1><br>
      <h2 class="demo-home__desc">
        2021-2022-2学期《机器学习》课程作业
      </h2>
    </div>
    <div class="upload">
      <h4 class='title'>拍照或上传图片</h4>
<!--      <van-uploader :after-read="afterRead" />-->
<!--      van-uploader file-list="{{ fileList }}" max-count="9" deletable="{{ true }}"  bind:after-read="afterRead"/>-->
      <van-uploader
        v-model="fileList"
        multiple
        :max-count="1"
        :after-read="afterRead"
        preview-size="200px"
        class="uploader"
        ref="file"
      /><br/>
      <van-button class='button' round  type="primary" size='noraml' @click='showPopUp'>识别</van-button>
    </div>
    <div class='popup'>
      <van-popup
        v-model:show="show"
        round
        closeable
        close-icon="close"
        position="bottom"
        :style="{ height: '40%' }"
      >
        <van-image
          class='pop_img'
          width="100"
          height="100"
          :src="img_show"
        />
        <van-cell-group>
          <van-cell title="识别结果" v-model:value="result" icon="search" />
        </van-cell-group>

      </van-popup>
    </div>

  </div>
</template>

<script>

import { Uploader } from 'vant'
import { Button } from 'vant';
import { Popup } from 'vant';
import { Image as VanImage } from 'vant';
import { Cell, CellGroup } from 'vant';
import { Notify } from 'vant';


import { image, uploadImage } from '@/api/user'
import axios from 'axios'

import { compressImage } from '@/assets/js/compressImage'// 图片压缩方法


export default {
  components: {
    [Uploader.name]: Uploader,
    [Button.name]: Button,
    [Popup.name]: Popup,
    [VanImage.name]: VanImage,
    [Cell.name]: Cell,
    [CellGroup.name]: CellGroup,
    [Notify.name]: Notify,
  },
  data() {
    return {
      fileList: [],//默认是一个空数组
      temp_img: '',
      imageData: {},
      isShow: false,
      showList: false,
      show: false,
      result: '121',
      img_url: '',
      img_show: ''

    };
  },

  computed: {},

  mounted() { },

  methods: {
    afterRead(file) {
      let config = {
        width: 224, // 压缩后图片的宽
        height: 224, // 压缩后图片的高
        quality: 0.9 // 压缩后图片的清晰度，取值0-1，值越小，所绘制出的图像越模糊
      }
      // 此时可以自行将文件上传至服务器
      console.log(file);

      //调用压缩方法
      compressImage(file.file, config).then(result => { // result 为压缩后二进制文件
        //返回压缩后的图片
        this.temp_img = result
        console.log("result:"+result)
      })

      //在这块创建FormData对象
      // FormData 对象的使用：
      // 1.用一些键值对来模拟一系列表单控件：即把form中所有表单元素的name与value组装成一个queryString
      // 2. 异步上传二进制文件。
      //上传图片需要转换二进制这里要用到FormData
      const forms = new FormData();
      //这里的file表示给后台传的属性名字，这里看接口文档需要传的的属性
      forms.append("file", file.file); // 获取上传图片信息
      console.log(forms)
      //向后端发送相应的请求
      //这块的url是具体的交互接口
      //headers是上传图片需要用到的响应头，此处的token是后端那边给设置的，所以我请求的时候需要带上token，
      //token根据实际情况自行添加
      axios
        .post('http://82.157.194.33:8002/upload_image', forms, {
          headers: {
            "content-type": "multipart/form-data"
          },
        })
        .then((res) => {
          //如果传入的响应状态码为200，则成功将文件发送给后台

          if (res.data.code == 200) {
            this.img_url = res.data.image_url_min

            Notify({ type: 'success', message: '图片上传成功' });
            //this.imageData = res.data.showapi_res_body;
            //this.isShow = false;
            //this.showList = true;
            //Toast(res.data.showapi_res_body.remark);
          } else {
            //Toast(res.data.msg);
            //this.isShow = false;
            console.log(res.data.msg)//这块是请求失败后台给返回的相应的数据
            Notify("图片上传错误")
          }
        });
    },
    showPopUp() {
      image(this.img_url).then((res) => {
        console.log(res)
        this.img_show = 'http://82.157.194.33:8002/image/' + this.img_url
        if (res.code == 200) {
          var classes = ['限速5m/s','限速15m/s','限速30m/s','限速40m/s','限速50m/s','限速60m/s','限速70m/s','限速80m/s','禁止直行和向左转弯','禁止直行和向右转弯','禁止直行',
                          '禁止向左转弯','禁止向左向右转弯','禁止向右转弯','禁止超车','禁止掉头','禁止机动车驶入','禁止鸣喇叭','解除限制速度40','解除限制速度50','直行和向右转弯',
                          '直行','向左转弯','向左和向右转弯','向右转弯','靠左侧道路行驶','靠右侧道路行驶','环岛行驶','机动车行驶','鸣喇叭','非机动车行驶',
                          '允许掉头','左右绕行','注意信号灯','注意危险','注意行人','注意非机动车','注意儿童','向右急弯路','向左急弯路','下陡坡',
                          '上陡坡','慢行','T形交叉','T形交叉','村庄','反向弯路','无人看守铁道路口','施工','连续弯路','有人看守铁道路口',
                          '事故易发路段','停车让行','禁止通行','禁止车辆临时或长时停放','禁止驶入','减速让行','停车检查']
          this.result = classes[res.classe]
          this.show =true;
        }
      })
    }
  }
}
</script>
<style lang="scss" scoped>
.index-container {
  .warpper {
    padding: 12px;
    background: #fff;
    .demo-home__title {
      margin: 0 0 6px;
      font-size: 32px;
      .demo-home__title img,
      .demo-home__title span {
        display: inline-block;
        vertical-align: middle;
      }
      img {
        width: 32px;
      }
      span {
        margin-left: 16px;
        font-weight: 500;
      }
    }
    .demo-home__desc {
      margin: 0 0 20px;
      color: rgba(69, 90, 100, 0.6);
      font-size: 14px;
    }
  }
  .upload {
    padding: 20px;
    background: #fff;
    text-align: center;
    .title {
      font-size: 15px;
      text-align: left;
    }
    .button {
      width: 95px;
      height: 35px;
    }
  }
  .popup {
    background: #fff;
    text-align: center;
    .pop_img {
      background: #fff;
      margin: 30px 30px 0 0 ;
      width: 100px;
      height: 100px;

    }
    .van-cell {
      margin: 20px 0 0 0 ;
      font-size: 15px;
      text-align: left;
    }
  }
}
</style>
