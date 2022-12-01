import numpy as np
import torch
import matplotlib.pyplot as plt



def visualization(img, gt_mask, mask, show_img=False, out_file=False):

    label2color_dict = [[106, 90, 205, 255], [73, 218, 154, 255], [235, 117, 50, 255],
                            [247, 208, 56, 255], [163, 224, 71, 255],
                            [52, 187, 230, 255], [67, 85, 219, 255],
                            [209, 58, 231, 255], [244, 58, 74, 255],
                            [244, 58, 200, 255], [44, 170, 155, 255], [255, 102, 153, 255]]
    
    img = img.numpy().astype(np.uint8)
    gt_mask = gt_mask.numpy().astype(np.uint8)
    # gt_mask = gt_mask.numpy().astype(np.uint8)[:, :, :, np.newaxis].repeat(4, axis=-1)
    mask = mask.numpy().astype(np.uint8)[:, :, :, np.newaxis].repeat(4, axis=-1)

    n = gt_mask.shape[0]

    for i in range(n):
        
        mask[i] = mask[i] * (np.array(label2color_dict[i]).astype(np.uint8)[np.newaxis, np.newaxis, :])
        # gt_mask[i] = gt_mask[i] * (np.array(label2color_dict[i]).astype(np.uint8)[np.newaxis, np.newaxis, :])

        if show_img:
            plt.axis('off')
            plt.subplot(1, 3, 1)
            plt.title('Image')
            plt.imshow(img)
            plt.subplot(1, 3, 2)
            plt.title('Pred')
            plt.imshow(img, alpha=1)
            plt.imshow(mask[i], alpha=0.5)
            plt.subplot(1, 3, 3)
            plt.title('Gt')
            plt.axis('off')
            # plt.imshow(img, alpha=0)
            plt.imshow(gt_mask[i], cmap='gray')
        else:
            plt.subplot(1, 2, 1)
            plt.title('Pred')
            plt.imshow(mask[i])
            plt.subplot(1, 2, 2)
            plt.title('Gt')
            plt.axis('off')
            plt.imshow(gt_mask[i])

        plt.show()


def visual_attn(img, attn, show_img=True, out_file=False):

    h, w = img.shape[0], img.shape[1]
    img = img.numpy().astype(np.uint8)
    point = 420

    attn = torch.from_numpy(attn)
    attn = attn[:, :, point, :].unsqueeze(dim=2).view(1, 8, 40, 40)

    attn = torch.nn.functional.interpolate(attn, (h, w), mode='bilinear')

    attn = attn.numpy()

    for i in range(8):

        if show_img:
            plt.axis('off')
            plt.title('Image')
            plt.imshow(img, alpha=1)
            plt.imshow(attn[0][i], alpha=0.5)

        plt.show()

def visual_overall(img, gt_mask, mask, show_img=False, out_file=False):

    label2color_dict = [[235, 117, 50, 255], [106, 90, 205, 255], [73, 218, 154, 255], 
                            [247, 208, 56, 255], [163, 224, 71, 255],
                            [52, 187, 230, 255], [67, 85, 219, 255],
                            [209, 58, 231, 255], [244, 58, 74, 255],
                            [244, 58, 200, 255], [44, 170, 155, 255], [255, 102, 153, 255]]
    
    pure_img = img.numpy().astype(np.uint8)
    img = np.dot(pure_img,[0.299,0.587,0.114])
    gt_mask = gt_mask.numpy().astype(np.uint8)[:, :, :, np.newaxis].repeat(4, axis=-1)
    mask = mask.numpy().astype(np.uint8)[:, :, :, np.newaxis].repeat(4, axis=-1)

    n = gt_mask.shape[0]


    if show_img:
        
        plt.axis('off')
        plt.title('Image')
        plt.imshow(pure_img, alpha=1)
        plt.show()
        
        plt.axis('off')
        plt.title('Pred')
        plt.imshow(img, alpha=1, cmap='gray')
        for i in range(n):
        
            mask[i] = mask[i] * (np.array(label2color_dict[i]).astype(np.uint8)[np.newaxis, np.newaxis, :])

            plt.imshow(mask[i], alpha=0.5)
        
        if out_file:
            plt.savefig('pred.jpg', dpi=300, bbox_inches='tight')
        plt.show()
        

        plt.axis('off')
        plt.title('Gt')
        plt.imshow(img, alpha=1, cmap='gray')
        print(n)
        for i in range(n):

            gt_mask[i] = gt_mask[i] * (np.array(label2color_dict[i]).astype(np.uint8)[np.newaxis, np.newaxis, :])
            plt.imshow(gt_mask[i], alpha=0.5)

        if out_file:
            plt.savefig('gt.jpg', dpi=300, bbox_inches='tight')
        plt.show()