# Beta-VAE

![CelebA_Reconstructions](https://github.com/TomBekor/BetaVAE/blob/master/figures/CelebA_reconstructions.png)

This repository's main notebook is `results.ipynb`.

Links to download the datasets:
1. [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. [Chairs Dataset](https://www.di.ens.fr/willow/research/seeing3Dchairs/)
3. [dSprites](https://github.com/deepmind/dsprites-dataset)
4. [FERG](http://grail.cs.washington.edu/projects/deepexpr/ferg-2d-db.html)

To train the models run `train.py`. <br>
Feel free to play with these parameters:<br>
`dataset_name` = `CelebA`/`Chairs`/`DSprites`/`FERG`<br>
`num_epochs = 10`<br>
`batch_size= 64`<br>
`trainset_percentage = 0.8`<br>
`betas = [1, 5, 25, 50, 100, 250, 500]`<br>
