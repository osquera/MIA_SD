import matplotlib.pyplot as plt
import numpy as np


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


x = np.array([0, 4, 8, 12, 16])
y = np.array([0.70, 0.87, 0.89, 0.85, 0.58])

fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
ax.plot(x, y, 'o-')
ax.set_xticks(x)
ax.set_ylim(0.5, 1)
ax.set(xlabel='Guidance scale $s$', ylabel='AUC Score',
       title='AUC Score vs. Guidance Scale')
fig.savefig("guidance_scale.pgf")
fig.savefig("guidance_scale.png")



