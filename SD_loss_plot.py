import pandas as pd
import matplotlib.pyplot as plt

plt.rc('legend', fontsize=30)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', titlesize=24)
plt.rc('figure', titlesize=30)

plt.rcParams.update({
    'figure.constrained_layout.use': True,
    "pgf.texsystem": "xelatex",
    "font.family": "serif", 
    'pgf.rcfonts': False, # Disables font replacement
    "pgf.preamble": "\n".join([
        r'\usepackage{mathtools}'
        r'\usepackage{fontspec}'
        r'\usepackage[T1]{fontenc}'
        r'\usepackage{kpfonts}'
        r'\makeatletter'
        r'\AtBeginDocument{\global\dimen\footins=\textheight}'
        r'\makeatother'
    ]),
})


df = pd.read_csv('dtu-400-target-loss.csv')
epochs = 400
steps_per_epoch = int(df.shape[0]/epochs)
fig = df.iloc[:,1].rolling(steps_per_epoch).mean()[steps_per_epoch-1::steps_per_epoch].reset_index().iloc[:,1].plot(title="Train loss (DTU)", xlabel="Epoch", ylabel="Loss")

fig.figure.savefig('dtu-400-target-loss.png')
fig.figure.savefig('images_attack_model/figures/dtu-400-target-loss.pgf')