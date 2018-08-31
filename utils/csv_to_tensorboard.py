import os
import argparse
from tensorboardX import SummaryWriter
import csv

def main():
    global args
    args = get_args()

    log_dir = os.path.join(args.result_path, 'log')
    csvfiles = os.listdir(log_dir)
    csvfiles = [os.path.join(log_dir,f) for f in csvfiles if f.endswith('csv')]
    csvfiles = csvfiles[0]

    train_loss = {}
    train_acc = {}
    val_loss = {}
    val_acc = {}

    epoch = 0
    with open(csvfiles, mode='r') as csv_file:
        # csv_reader = csv.DictReader(csv_file)
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            train_parse_str = '[%d]'%(epoch) + args.train_parser
            if(train_parse_str in row[0]):
                train_loss[str(epoch)] = float(row[3][row[3].find('(')+1:row[3].find(')')])
                train_acc[str(epoch)] = float(row[4][row[4].find('(')+1:row[4].find(')')])
                epoch += 1

            if(args.val_parser in row[0]):
                val_loss[str(epoch)] = float(row[2][row[2].find('(')+1:row[2].find(')')])
                val_acc[str(epoch)] = float(row[3][row[3].find('(')+1:row[3].find(')')])

    writer = SummaryWriter(args.result_path)

    for epoch in range(args.epoch):
        num_iter = (epoch + 1) * (args.num_dataset)
        writer.add_scalar('train_loss', train_loss[str(epoch)], num_iter)
        writer.add_scalar('train_acc_top1', train_acc[str(epoch)], num_iter)
        try:
            writer.add_scalar('val_loss', val_loss[str(epoch)], num_iter)
            writer.add_scalar('val_acc_top1', val_acc[str(epoch)], num_iter)
        except:
            pass
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--result_path', type=str, help="path where metadata stored")
    parser.add_argument('-n', '--num_dataset', type=int, help="number of training datasets") # v1 (86017, 11522)
    parser.add_argument('-v', '--version', type=int, help="version of sthsth datasets")
    parser.add_argument('-e', '--epoch', type=int, help="final epoch")
    parser.add_argument('--train_parser', type=str, help="str to parse train") # [d][2860/2868]
    parser.add_argument('--val_parser', type=str, help="str to parse val") # [380/385]


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()