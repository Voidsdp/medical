import torchvision.utils as vutils

def generate_img(G,label,result_dir):
    fake_image = G(label)
    vutils.save_image(fake_image,result_dir)
