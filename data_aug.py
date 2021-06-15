from PIL import Image, ImageOps, ImageFilter, ImageEnhance


import numpy as np
import random

generated_gt_file = open("train_gt_generated.txt", "w+")


with open("./train_gt.txt", 'r') as file:
    for info in file:
        # originial file
        generated_gt_file.write(info)

        data_augmentation = random.random()
        if (data_augmentation < 0.3):
            
            info_tmp = info.strip(' ').split()
            img_file = "." + info_tmp[0][0:]
            label_instance_file = "." + info_tmp[1][0:]
            label_existence = np.array([int(info_tmp[2]), int(info_tmp[3]), int(info_tmp[4]), int(info_tmp[5])])   

            img_name = info_tmp[0][-9:]
            image_path = "." + info_tmp[0][:-9]

            lable_img_name = info_tmp[1][-9:]
            lable_imgae_path = "." + info_tmp[1][:-9]

            
            img = Image.open(img_file)
            lable_img = Image.open(label_instance_file)

            save_img_name = ""
            save_label_img_name = ""
            generated_info = ""

            aug_type = random.random()
            
            if aug_type < 0.4:
                # darken file
                save_img_name = "1" + img_name[1:]
                save_label_img_name = "1" + lable_img_name[1:]

                enhancer = ImageEnhance.Brightness(img)
                img_dark = enhancer.enhance(0.75)
                img_dark.save(image_path + save_img_name)
                lable_img.save(lable_imgae_path + save_label_img_name)

                generated_info = image_path[1:] + save_img_name + " " + lable_imgae_path[1:] + save_label_img_name
                for i in range(4):
                    generated_info += " "
                    generated_info += info_tmp[5 - i]

            elif aug_type >= 0.4 and aug_type < 0.7:
                # guassian blur
                save_img_name = "2" + img_name[1:]
                save_label_img_name = "2" + lable_img_name[1:]

                img_blur = img.filter(ImageFilter.GaussianBlur(1.2))
                img_blur.save(image_path +  save_img_name)
                lable_img.save(lable_imgae_path + save_label_img_name)

                generated_info = image_path[1:] + save_img_name + " " + lable_imgae_path[1:] + save_label_img_name
                for i in range(4):
                    generated_info += " "
                    generated_info += info_tmp[2 + i]

            else:
                # decrease contrast
                save_img_name = "3" + img_name[1:]
                save_label_img_name = "3" + lable_img_name[1:]

                enhancer = ImageEnhance.Contrast(img)
                img_contrast = enhancer.enhance(0.75)
                img_contrast.save(image_path +  save_img_name)
                lable_img.save(lable_imgae_path + save_label_img_name)

                generated_info = image_path[1:] + save_img_name + " " + lable_imgae_path[1:] + save_label_img_name
                for i in range(4):
                    generated_info += " "
                    generated_info += info_tmp[5 - i]
            generated_gt_file.write(generated_info + "\n")