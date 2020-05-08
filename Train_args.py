import argparse


def get_input_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', type=str, default='ImageClassifier/flowers', 
                        help='path to folder of images')
    
    parser.add_argument('--dir_image', type=str, default='ImageClassifier/flowers/test/2/image_05133.jpg', 
                        help='path to folder to predicted image')
    
    parser.add_argument('--arch', type=str, default='vgg16', 
                        help='model densetnet121 or vgg16')   
    
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='Path save checkpoints')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs')
    
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='number of hidden units')
    
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='dropping rate')
    
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='GPU')
    
    parser.add_argument('--load_dir', type=str, default='checkpoint.pth',
                        help='Path load checkpoint')
    
    parser.add_argument('--nb_topk', type=int, default=5,
                        help='Number of class')
    
    

    
    

    return  parser.parse_args() 