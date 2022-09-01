import model as modelAuto
import argparse


def main():
    opt = parser.parse_args()
    AUtoencoder = modelAuto.AutoEncorder()
    AUtoencoder.train(opt.train_path, opt.test_path, int(opt.epochs), int(opt.batch_size))
    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(prog='main.py', description='')
    parser.add_argument('-train', '--train_path', type=str, default='data/train', help='train data path')
    parser.add_argument('-test', '--test_path', type=str, default='data/test', help='test data path')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='size of each image batch')
    
    main()