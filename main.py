
from utils.config import Config


def main():
    cfgs = Config()

    if cfgs.dataset == "AV-MNIST":
        from run.AVMNIST_main import AVMNIST_main
        AVMNIST_main(cfgs)

    elif cfgs.dataset == "CREMAD":
        from run.CREMAD_main import CREMAD_main
        CREMAD_main(cfgs)

    elif cfgs.dataset == "URFunny":
        from run.URFunny_main import URFunny_main
        URFunny_main(cfgs)

    elif cfgs.dataset == "AVE":
        from run.AVE_main import AVE_main
        AVE_main(cfgs)

    elif cfgs.dataset == "MOSEI":
        from run.MOSEI_main import MOSEI_main
        MOSEI_main(cfgs)


if __name__ == '__main__':
    main()
