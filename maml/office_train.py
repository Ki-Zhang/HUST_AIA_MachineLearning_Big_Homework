import  torch, os
import  numpy as np
from    office import Office
# import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from torch.utils.tensorboard import SummaryWriter


from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    writer = SummaryWriter('/root/tf-logs')
    t_sim = 0
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # config = [
    #     ('conv2d', [32, 3, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 1, 0]),
    #     ('flatten', []),
    #     ('linear', [args.n_way, 32 * 5 * 5])
    # ]

    # Resnet-18
    config = [
        # conv1
        ('conv2d', [64, 3, 7, 7, 2, 3]),
        ('bn', [64]),
        ('max_pool2d', [3, 2, 1]),

        # conv2-1
        ('res18_up', [64, 64, 3, 3, 1, 1]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('res_add', []),
        # conv2-2
        ('res18_up', [64, 64, 3, 3, 1, 1]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('relu', [True]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('bn', [64]),
        ('res_add', []),

        # conv3-1
        ('res18_up', [128, 64, 1, 1, 2, 0]),
        ('conv2d', [128, 64, 3, 3, 2, 1]),
        ('bn', [128]),
        ('relu', [True]),
        ('conv2d', [128, 128, 3, 3, 1, 1]),
        ('bn', [128]),
        ('res_add', []),
        # conv3-2
        ('res18_up', [128, 128, 3, 3, 1, 1]),
        ('conv2d', [128, 128, 3, 3, 1, 1]),
        ('bn', [128]),
        ('relu', [True]),
        ('conv2d', [128, 128, 3, 3, 1, 1]),
        ('bn', [128]),
        ('res_add', []),

        # conv4-1
        ('res18_up', [256, 128, 1, 1, 2, 0]),
        ('conv2d', [256, 128, 3, 3, 2, 1]),
        ('bn', [256]),
        ('relu', [True]),
        ('conv2d', [256, 256, 3, 3, 1, 1]),
        ('bn', [256]),
        ('res_add', []),
        # conv4-2
        ('res18_up', [256, 256, 3, 3, 1, 1]),
        ('conv2d', [256, 256, 3, 3, 1, 1]),
        ('bn', [256]),
        ('relu', [True]),
        ('conv2d', [256, 256, 3, 3, 1, 1]),
        ('bn', [256]),
        ('res_add', []),

        # conv5-1
        ('res18_up', [512, 256, 1, 1, 2, 0]),
        ('conv2d', [512, 256, 3, 3, 2, 1]),
        ('bn', [512]),
        ('relu', [True]),
        ('conv2d', [512, 512, 3, 3, 1, 1]),
        ('bn', [512]),
        ('res_add', []),
        # conv5-2
        ('res18_up', [512, 512, 3, 3, 1, 1]),
        ('conv2d', [512, 512, 3, 3, 1, 1]),
        ('bn', [512]),
        ('relu', [True]),
        ('conv2d', [512, 512, 3, 3, 1, 1]),
        ('bn', [512]),
        ('res_add', []),

        ('avg_pool2d', [7, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 512*1*1])
    ]

    normal_amazon = [[0.7921833, 0.7860339, 0.7840052], [0.250673, 0.25547105, 0.25698686]]
    normal_dslr = [[0.47089794, 0.44867676, 0.40638262], [0.16405636, 0.15768151, 0.16018835]]
    normal_webcam = [[0.61201507, 0.618804, 0.61735976], [0.20979871, 0.21559623, 0.21876234]]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = Office(root='../data_set/office_31'
                  , name='amazon'
                  , n_way=args.n_way
                  , k_shot=args.k_spt
                  , k_query=args.k_qry
                  , batchsz=10000
                  , resize=args.imgsz
                  , normalize=normal_amazon)
    mini_test = Office(root='../data_set/office_31'
                       , name='dslr'
                       , n_way=args.n_way
                       , k_shot=args.k_spt
                       , k_query=args.k_qry
                       , batchsz=100
                       , resize=args.imgsz
                       , normalize=normal_dslr)
    # mini_test2 = Office(root='/home/kdzhang/PC&ML/my_maml/office'
    #                    , name='webcam'
    #                    , n_way=args.n_way
    #                    , k_shot=args.k_spt
    #                    , k_query=args.k_qry
    #                    , batchsz=100
    #                    , resize=args.imgsz
    #                    , normalize=normal_webcam)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=4, pin_memory=False)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)
            t_sim += 1
            writer.add_scalar('train-acc', accs[-1], t_sim)
            if step % 50 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=4, pin_memory=False)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                
                writer.add_scalar('test-acc', accs[-1], t_sim)
                print('Test acc:', accs)

    torch.save(maml.state_dict(), 'maml.pt')
    # m_state_dict = torch.load('maml.pt')
    # new_maml = Meta(args, config).to(device)
    # new_maml.load_state_dict(m_state_dict)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=80000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    # argparser.add_argument('--n_way', type=int, help='n way', default=31)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=2)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=3)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=224)
    # argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)      # 表示图片的通道数
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=7)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
