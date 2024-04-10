## 背景

这个项目可以作为benchmark 的一个轻量级替代。benchmark常年面临着机器不足、资源紧张、无法自定义拓展的机器的问题。开发者手里有一些32G的V100和A100无法被自由的加入到benchmark队列中。为了解决这个问题，提高迭代的效率，我开发了这个项目。**只需要一次benchmark运行，进行一些准备措施，就可以随时使用自己的资源进行benchmark一样的配置进行执行和性能测试。方便开发人员的性能优化。

## 目录介绍

```
train_single_config.py      入口脚本，可以进行一个配置的训练和ips统计操作。一键集成。只需要传入套件名称和配置名称
detect_command.sh           deprecated.
get_run_command.py          获取 benchmark 某个配置的执行命令。作为 train_single_config.py 的一个组件使用，不建议单独使用
logs/                       临时存放某些中间文件的目录
run_model_sh.log            必要，第一次benchmark并使用console获取的配置文件，被get_run_command.py文件使用，用来获取训练命令的。
```

## 使用方法

#### Step1：执行一次benchmark并获取run_model_sh.log

运行一次 benchmark 并从提供的 `dynamicTostatic/run_model_sh` 网站下爬取所有的配置，并创建此文件。

文件格式大概如下：
```
http://wangying28-2020-004-1.bcc-bdbl.baidu.com:8666/PaddleSingleMachineLog/paddle2023_09_23_06_26_39_V100_11.7_8.4.1_develop_3.10/dynamicTostatic/run_model_s
h/PaddleClas_AlexNet_bs64_fp16_DP_N1C1_d2sT_ModelRunTemp.sh
http://wangying28-2020-004-1.bcc-bdbl.baidu.com:8666/PaddleSingleMachineLog/paddle2023_09_23_06_26_39_V100_11.7_8.4.1_develop_3.10/dynamicTostatic/run_model_s
h/PaddleClas_AlexNet_bs64_fp16_DP_N1C8_d2sT_ModelRunTemp.sh
http://wangying28-2020-004-1.bcc-bdbl.baidu.com:8666/PaddleSingleMachineLog/paddle2023_09_23_06_26_39_V100_11.7_8.4.1_develop_3.10/dynamicTostatic/run_model_s
h/PaddleClas_AlexNet_bs64_fp32_DP_N1C1_d2sT_ModelRunTemp.sh
http://wangying28-2020-004-1.bcc-bdbl.baidu.com:8666/PaddleSingleMachineLog/paddle2023_09_23_06_26_39_V100_11.7_8.4.1_develop_3.10/dynamicTostatic/run_model_s
h/PaddleClas_AlexNet_bs64_fp32_DP_N1C8_d2sT_ModelRunTemp.sh
http://wangying28-2020-004-1.bcc-bdbl.baidu.com:8666/PaddleSingleMachineLog/paddle2023_09_23_06_26_39_V100_11.7_8.4.1_develop_3.10/dynamicTostatic/run_model_s
h/PaddleClas_CLIP_vit_base_patch16_224_bs128_amp_DP_N1C1_d2sT_ModelRunTemp.sh
http://wangying28-2020-004-1.bcc-bdbl.baidu.com:8666/PaddleSingleMachineLog/paddle2023_09_23_06_26_39_V100_11.7_8.4.1_develop_3.10/dynamicTostatic/run_model_s
h/PaddleClas_CLIP_vit_base_patch16_224_bs128_amp_DP_N1C8_d2sT_ModelRunTemp.sh
```

就是网站上所有配置的 train_sh 的 href 地址。具体的提取脚本 javascript 脚本如下：

```
# 在 chrome 的 F12 开发者模式的 console 下运行这部分代码，支持 JQuery
var importJs=document.createElement('script');
importJs.setAttribute("type","text/javascript");
importJs.setAttribute("src", 'https://code.jquery.com/jquery-1.12.4.min.js');
document.getElementsByTagName("head")[0].appendChild(importJs);

# 运行下面代码
tmp = $("a");
for (var i=0;i<$("a").length;i++) {
    console.log(tmp[i].href);
}

```

执行了上述代码之后，将文件导出为 log 文件，使用vim打开，修改为上述的格式。可以使用块编辑。

#### Step2：启动运行

```
cat /root/PaddleSeg/to_train.txt | xargs -n1 -I {} python /root/auto_command/train_single_config.py --name {} --repo PaddleSeg 2>error_seg.txt 1>output_seg.txt && cat /root/PaddleClas/to_train.txt | xargs -n1 -I {} python /root/auto_command/train_single_config.py --name {} --repo PaddleClas 2>error_clas.txt 1>output_clas.txt
```

#### Step3：数据统计

Step2 中我们其实将所有的输出都写入到了 output_seg.txt 中，这里我们来解析这部分数据。

```
python analysis_log.py --input output_seg.txt --output output_seg.png
python analysis_log.py --input output_seg.txt --output output_seg.xlsx

python analysis_log.py --input output_clas.txt --output output_clas.png
python analysis_log.py --input output_clas.txt --output output_clas.xlsx
```

rm -rf /home/data/datas/*
cp ./*.png /home/data/datas
cp ./*.xlsx /home/data/datas



python train_single_config.py --name xxx --profile