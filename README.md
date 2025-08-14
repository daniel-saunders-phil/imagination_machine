To maintain the website, make sure you are always on the `main` branch, never `gh-pages` branch.

Edit your blog posts as `.qmd` files in the `posts` directory. You can ignore the `_site` and `_freeze`

You can locally preview the website with:

```bash
quarto preview
```

You can publish the website with:

```bash
quarto publish gh-pages
```

Ideas for forthcoming blog posts:

- Multi-modality in MMMs
- four way correlations in multi-channel MMMs
- Penalized complexity as a framework for MMMs
- - Saturation priors that reduce to linear regressions in the absence of information
- - Adstock that reduces to linear regression in the absence of information
- Marginal effects for interpreting co-efficients
- Centered / non-centered reparameterization tricks for positively constrained parameters.
- When exchangeability fails
- Centered / non-centered reparameterization tricks for 0-1 constrained parameters.
- Partially centered and non-centered parameterizations
- Zero-sum transformation