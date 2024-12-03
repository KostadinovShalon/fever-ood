from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 18})


def plot_ablation(model_name, dataset, weights, fpr95, auroc, min_fpr95=20, max_fpr95=50, min_auroc=85, max_auroc=97):
    series = ['LSV Reg', 'CN Reg']

    fig, ax1 = plt.subplots()
    ax1.plot(weights, fpr95[0], label=series[0], color='blue', marker='o')
    ax1.plot(weights, fpr95[1], label=series[1], color='blue', marker='^', linestyle='--')
    ax1.set_ylabel('FPR95', color='blue')
    ax1.set_xlabel('λ')
    ax1.set_ylim(min_fpr95, max_fpr95)
    # Add right axis
    ax2 = ax1.twinx()
    ax2.plot(weights, auroc[0], label=series[0], color='red', marker='o', linestyle='-')
    ax2.plot(weights, auroc[1], label=series[1], color='red', marker='^', linestyle='--')
    ax2.set_ylabel('AUROC', color='red')
    ax2.set_ylim(min_auroc, max_auroc)

    plt.title(f'{model_name} on {dataset} as ID')
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(f'ablation_{model_name}_{dataset}.png')
    plt.show()


###### Dream-OOD-128-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[43.48, 43.51, 45, 46.78, 100],
         [43.48, 51.12, 48.01, 100, 100]]
auroc = [[89.092, 88.184, 88.6, 89.362, 50],
         [89.092, 87.554, 88.444, 50, 50]]

###### VOS CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[33.30, 32.51, 34.62, 30.59, 29.16],
         [33.3, 34.48, 37.37, 31.22, 39.87]]
auroc = [[92.17, 92.49, 91.96, 93.41, 93.51],
         [92.17, 91.81, 90.78, 92.20, 90.20]]

plot_ablation(model_name='VOS',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### VOS-96-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[29.89, 39.38, 31.90, 33.29, 25.48],
         [29.89, 30.63, 35.92, 29.50, 100.00]]
auroc = [[93.98, 91.84, 93.59, 93.50, 95.36],
         [93.98, 93.87, 92.66, 94.20, 50.00]]

plot_ablation(model_name='VOS-96-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### VOS-64-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[31.20, 30.80, 33.62, 32.46, 27.92],
         [31.20, 30.72, 30.24, 33.97, 100.00]]
auroc = [[93.44, 94.09, 93.63, 93.84, 94.70],
         [93.44, 93.93, 93.86, 93.49, 50.00]]

plot_ablation(model_name='VOS-64-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### VOS-32-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[35.65, 31.17, 31.47, 35.70, 35.53],
         [35.65, 33.04, 37.27, 29.60, 100.00]]
auroc = [[93.07, 93.91, 93.82, 92.49, 92.78],
         [93.07, 93.07, 93.27, 93.78, 50.00]]

plot_ablation(model_name='VOS-32-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### VOS-10-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[37.18, 41.85, 35.99, 100.00, 100.00],
         [37.18, 45.33, 47.82, 100.00, 100.00]]
auroc = [[93.14, 92.39, 92.59, 50.00, 50.00],
         [93.14, 90.86, 86.62, 50.00, 50.00]]

plot_ablation(model_name='VOS-10-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### FFS CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[33.96, 25.24, 32.81, 36.54, 32.49],
         [33.96, 36.67, 40.83, 34.01, 26.95]]
auroc = [[91.57, 93.41, 92.14, 91.89, 93.30],
         [91.57, 90.43, 91.66, 90.73, 93.18]]

plot_ablation(model_name='FFS',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### FFS-96-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[27.96, 29.26, 24.34, 27.61, 24.93],
         [27.96, 28.26, 34.05, 34.62, 100.00]]
auroc = [[94.22, 93.85, 94.85, 94.32, 95.26],
         [94.22, 94.37, 92.84, 93.06, 50.00]]

plot_ablation(model_name='FFS-96-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### FFS-64-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[33.92, 28.29, 32.80, 27.30, 24.93],
         [33.92, 24.16, 30.53, 31.95, 100.00]]
auroc = [[92.44, 94.23, 93.37, 94.66, 94.84],
         [92.44, 94.67, 93.68, 92.88, 50.00]]

plot_ablation(model_name='FFS-64-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### FFS-32-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[30.77, 28.01, 35.27, 32.29, 36.86],
         [30.77, 29.86, 28.64, 25.95, 100.00]]
auroc = [[93.09, 94.19, 92.78, 93.00, 92.66],
         [93.09, 93.28, 94.31, 94.14, 50.00]]

plot_ablation(model_name='FFS-32-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

###### FFS-10-NSR CIFAR-10 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[38.48, 34.33, 44.61, 100.00, 100.00],
         [38.48, 41.67, 42.81, 100.00, 100.00]]
auroc = [[90.25, 92.78, 88.23, 50.00, 50.00],
         [90.25, 89.48, 88.40, 50.00, 50.00]]

plot_ablation(model_name='FFS-10-NSR',
              dataset='CIFAR-10',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc)

########################## CIFAR - 100 ##########################
###### VOS CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[72.44, 68.93, 72.74, 67.71, 69.18],
         [72.44, 79.00, 72.03, 100.00, 100.00]]
auroc = [[80.78, 82.22, 81.04, 83.24, 82.67],
         [80.78, 76.82, 80.64, 50.00, 50.00]]

plot_ablation(model_name='VOS',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=60,
              max_fpr95=80,
              min_auroc=75,
              max_auroc=85)

###### VOS-96-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[72.56, 67.20, 63.51, 69.08, 100.00],
         [72.56, 70.45, 73.59, 100.00, 100.00]]
auroc = [[80.45, 84.37, 84.70, 81.75, 50.00],
         [80.45, 82.30, 81.37, 50.00, 50.00]]

plot_ablation(model_name='VOS-114-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=60,
              max_fpr95=80,
              min_auroc=75,
              max_auroc=85)

###### VOS-64-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[73.02, 68.17, 100.00, 100.00, 100.00],
         [73.02, 100.00, 100.00, 100.00, 100.00]]
auroc = [[80.88, 81.70, 50.00, 50.00, 50.00],
         [80.88, 50.00, 50.00, 50.00, 50.00]]

plot_ablation(model_name='VOS-100-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=60,
              max_fpr95=80,
              min_auroc=75,
              max_auroc=85)

###### FFS CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[72.59, 72.57, 66.09, 70.65, 68.53],
         [72.59, 68.46, 74.14, 100.00, 100.00]]
auroc = [[79.80, 81.52, 83.66, 80.73, 81.99],
         [79.80, 82.96, 79.38, 50.00, 50.00]]

plot_ablation(model_name='FFS',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=60,
              max_fpr95=80,
              min_auroc=75,
              max_auroc=85)

###### FFS-96-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[71.28, 65.60, 67.70, 67.96, 100.00],
         [71.28, 71.68, 75.07, 100.00, 100.00]]
auroc = [[80.93, 84.25, 82.71, 81.46, 50.00],
         [80.93, 81.03, 79.89, 50.00, 50.00]]

plot_ablation(model_name='FFS-114-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=60,
              max_fpr95=80,
              min_auroc=75,
              max_auroc=85)

###### FFS-64-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[70.14, 69.69, 100.00, 100.00, 100.00],
         [70.14, 100.00, 100.00, 100.00, 100.00]]
auroc = [[81.24, 81.58, 50.00, 50.00, 50.00],
         [81.24, 50.00, 50.00, 50.00, 50.00]]

plot_ablation(model_name='FFS-100-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=60,
              max_fpr95=80,
              min_auroc=75,
              max_auroc=85)

###### Dream-OOD CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[50.96, 47.3, 44.33, 46.83, 47.57],
         [50.96, 48.5, 53.76, 53.57, 64.6]]
auroc = [[79.80, 81.52, 83.66, 80.73, 81.99],
         [79.80, 82.96, 79.38, 50.00, 50.00]]

plot_ablation(model_name='Dream-OOD',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=40,
              max_fpr95=65,
              min_auroc=65,
              max_auroc=92)

###### Dream-OOD-256-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[44.85, 46.51, 47.00, 47.50, 45.26],
         [44.85, 46.25, 49.47, 47.25, 100.00]]
auroc = [[88.06, 89.402, 89.83, 88.93, 88.78],
         [88.06, 88.944, 87.872, 87.098, 66.092]]

plot_ablation(model_name='Dream-OOD-256-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=40,
              max_fpr95=65,
              min_auroc=65,
              max_auroc=92)

###### Dream-OOD-128-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[43.48, 43.51, 45, 46.78, 100],
         [43.48, 51.12, 48.01, 100, 100]]
auroc = [[89.092, 88.184, 88.6, 89.362, 50],
         [89.092, 87.554, 88.444, 50, 50]]

plot_ablation(model_name='Dream-OOD-128-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=40,
              max_fpr95=65,
              min_auroc=65,
              max_auroc=92)

###### Dream-OOD-100-NSR CIFAR-100 ######
lambdas = ['0', '0.001', '0.01', '0.1', '1']
fpr95 = [[42.77, 66.15, 100, 100, 100],
         [42.77, 100, 100, 100, 100]]
auroc = [[89.976, 72.656, 50, 50, 50],
         [89.976, 50, 50, 50, 50]]

plot_ablation(model_name='Dream-OOD-100-NSR',
              dataset='CIFAR-100',
              weights=lambdas,
              fpr95=fpr95,
              auroc=auroc,
              min_fpr95=40,
              max_fpr95=65,
              min_auroc=65,
              max_auroc=92)
