import argparse
from generate_data_loader import  get_loader
from PIL import Image
import numpy as np
from network import U_Net
from utils import utils
import torch

def main(config):
    device = torch.device('cuda:0')
    model = U_Net(img_ch=1, output_ch=1).to(device)

    checkpoint = './models/U-Net.pkl'
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()

        
    test_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train')

    with torch.no_grad():
        for index, samper in enumerate(test_loader):
            print('process image' + str(index))
            #data_dict = test_loader.dataset.image_paths[index]
            img = samper['img'].to(device)
            #img = img.unsqueeze(0)
            pred_river = model(img)

            # to numpy arrays
            pred_river = pred_river.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)

            # save results
            img_name = samper['img_name'][0]

            # 1. save input image
            #img = utils.unnormalize(img[0, ...])
            #img = Image.fromarray(img[:,:,0])

            pred_river = np.clip(pred_river/20, 0, 1) * 65535.0
            pred_river = pred_river[0,:,:,0]
            height = Image.fromarray(pred_river.astype(np.uint16))
            target_path = './generate/results/' + img_name
            height.save(target_path, bits=16)
            print('process_image:%s' %(img_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./generate/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
