# Autoencoders

We use the autoencoder introduced in [^1]. We assume an orthogonal matrix ``\boldsymbol{U}\in\mathbb{R}^{n\times\nu}`` (``\boldsymbol{U}^{T}\boldsymbol{U}=\boldsymbol{I}``), a polynomial ``\boldsymbol{W}:\mathbb{R}^\nu\to \mathbb{R}^n`` that starts with quadratic terms up to order ``d``. The encoder is the linear map ``\boldsymbol{U}^T`` and the decoder is
```math 
\boldsymbol{W}\left(\boldsymbol{z}\right)=\boldsymbol{U}\boldsymbol{z}+\boldsymbol{W}\left(\boldsymbol{z}\right).
```

To find the autoencoder we first solve
```math
\arg\min_{\boldsymbol{U},\boldsymbol{W}}\sum_{k=1}^{N}\left\Vert \boldsymbol{y}_{k}\right\Vert ^{-2}\left\Vert \boldsymbol{W}\left(\boldsymbol{U}\left(\boldsymbol{y}_{k}\right)\right)-\boldsymbol{y}_{k}\right\Vert ^{2}.
```
then solve
```math
\arg\min_{\boldsymbol{S}}\sum_{k=1}^{N}\left\Vert \boldsymbol{x}_{k}\right\Vert ^{-2}\left\Vert \boldsymbol{W}\left(\boldsymbol{S}\left(\boldsymbol{U}\left(\boldsymbol{x}_{k}\right)\right)\right)-\boldsymbol{y}_{k}\right\Vert ^{2}.
```

The methods are

```@docs
AENCManifold
```

```@docs
zero(M::AENCManifold)
```

```@docs
AENCIndentify
```

[^1]: M. Cenedese, J. Axås, B. Bäuerlein, K. Avila, and G. Haller. Data-driven modeling and prediction of non-linearizable dynamics via spectral submanifolds. Nat Commun, 13(872), 2022.
