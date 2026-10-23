# Project website

Static research page for the Multi-View In-Cabin Dataset. The page links to the
paper, Hugging Face dataset and viewer, repository, and tutorial. Sensor illustrations are schematics; the dataset viewer links to the actual
released data.

Enable it in **Settings → Pages → Build and deployment**:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/docs**

The site will publish at:
https://evgenygorelik.github.io/multiview_incabin_dataset/

Edit `index.html` and `styles.css` to update the page. No build step or JavaScript
framework is needed. `.nojekyll` makes GitHub serve the static files directly.
