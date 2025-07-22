import time
import datetime
import jittor as jt

from Src.PFNet import PFNet
from Src.misc import check_mkdir
from jittor import transform
from collections import OrderedDict
from config import *
from PIL import Image
import numpy as np


jt.set_global_seed(2021)  # 设置随机种子
jt.flags.use_cuda = 1  # 使用CUDA


model_path = ('ckpt_pytorch/PFNet/PFNet.pth')




results_path = 'results/'
check_mkdir(results_path)
exp_name = 'PFNet'
args = {
    'scale': 416,
    'save_results': True
}

img_transform = transform.Compose([
    transform.Resize((args['scale'], args['scale'])),
    transform.ToTensor(),
    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


to_pil = transform.ToPILImage()

to_test = OrderedDict([
        ('CHAMELEON', chameleon_path),
        ('CAMO', camo_path),
        ('COD10K', cod10K_path)
])

results = OrderedDict()



def main():

    net = PFNet('Src/backbone/resnet/resnet50.pkl')

    net.load_state_dict(jt.load(model_path))

    print('Load {} succeed!'.format(model_path))


    net.eval()
    with jt.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            # 修改后（按文件名排序）
            img_list = [os.path.splitext(f)[0] for f in sorted(os.listdir(image_path)) if f.endswith('.jpg')]
            for idx, img_name in enumerate(img_list):
                #print(img_name)
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size


                transformed_img = img_transform(img)


                img_var = jt.array(transformed_img).unsqueeze(dim=0)



                start_each = time.time()
                _, _, _, prediction = net(img_var)

                time_each = time.time() - start_each
                time_list.append(time_each)

                # 分析transform.Toimage源码可知,如果输入Var参数为[C,H,W],因为只接受[H,W,C]的形式,
                #这就导致了使用默认的mode = 'RGB', 变为[C,H,3],最终经过导致输出为[ H , W,3]
                pred_cpu = jt.cpu(prediction.detach().squeeze(0))  # [1, 416, 416]

                # 转换维度：CHW -> HWC
                pred_hwc = pred_cpu.transpose(1, 2, 0) # [1,416,416] -> [416,416,1]

                #ToPILimage不支持单通道的浮点数还原
                pred_uint8 = (pred_hwc * 255).astype(np.uint8)
                prediction = np.array(transform.Resize((h, w))(to_pil(pred_uint8)))
                #print(prediction.shape)

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(os.path.join(results_path, exp_name, name, img_name + '.png'))

            print('{}'.format(exp_name))

            print("{}'s average Time Is : {:.3f} s".format(name, np.mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1/np.mean(time_list)))

    end = time.time()
    print("Total Testing Time : {}".format(str(datetime.timedelta(seconds=int(end-start)))))

if __name__ == '__main__':
    main()