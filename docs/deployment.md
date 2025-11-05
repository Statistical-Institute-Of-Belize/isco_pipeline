# Documentation Deployment

This guide explains how to build and deploy the documentation website.

## Local Development

### Install Documentation Dependencies

```bash
pip install -r requirements-docs.txt
```

### Build Documentation Locally

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

### Serve Documentation Locally

```bash
mkdocs serve
```

Then visit `http://127.0.0.1:8000` in your browser. The site will automatically reload when you save changes to the documentation files.

### Preview Options

- **`mkdocs serve`** — Run local development server with auto-reload
- **`mkdocs serve -a 0.0.0.0:8080`** — Serve on a different host/port
- **`mkdocs build --clean`** — Clean build (removes previous site/ directory)

## GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

### Automatic Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) that:

1. Triggers on every push to `main`
2. Installs MkDocs and dependencies
3. Builds the documentation
4. Deploys to the `gh-pages` branch

### Enable GitHub Pages

To enable GitHub Pages for the first time:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **Deploy from a branch**
4. Select the **gh-pages** branch and **/ (root)** folder
5. Click **Save**

After the first deployment, your documentation will be available at:

```
https://gianaguilar.github.io/isco_pipeline/
```

### Manual Deployment

You can also deploy manually from your local machine:

```bash
mkdocs gh-deploy
```

This command:
- Builds the documentation
- Pushes the `site/` contents to the `gh-pages` branch
- Updates GitHub Pages

!!! warning "Repository Permissions"
    Manual deployment requires push access to the repository and the ability to create/update the `gh-pages` branch.

### Deployment Workflow

```mermaid
graph LR
    A[Push to main] --> B[GitHub Actions Trigger]
    B --> C[Install Dependencies]
    C --> D[Build with MkDocs]
    D --> E[Deploy to gh-pages]
    E --> F[GitHub Pages Update]
    F --> G[Live Documentation]
```

## Customizing the Documentation

### Site Configuration

Edit `mkdocs.yml` to customize:

- **Site metadata** — name, description, author
- **Theme settings** — colors, fonts, features
- **Navigation structure** — add/remove/reorder pages
- **Markdown extensions** — code highlighting, admonitions, etc.

### Adding New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add the page to the `nav` section in `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - User Guide: users.md
  - Developer Guide: developers.md
  - API Reference: api.md
  - Configuration: configuration.md
  - Your New Page: new-page.md  # Add here
```

3. Commit and push — the page will be automatically deployed

### Styling and Theming

The documentation uses the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

Customize colors in `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: indigo  # Change primary color
    accent: indigo   # Change accent color
```

Available colors: red, pink, purple, deep purple, indigo, blue, light blue, cyan, teal, green, light green, lime, yellow, amber, orange, deep orange

## Troubleshooting

### Build Failures

If the documentation fails to build:

1. Check the GitHub Actions log in the **Actions** tab
2. Run `mkdocs build` locally to see detailed error messages
3. Common issues:
   - Broken internal links
   - Invalid YAML in `mkdocs.yml`
   - Missing dependencies

### Local Preview Issues

If `mkdocs serve` doesn't work:

- Ensure dependencies are installed: `pip install -r requirements-docs.txt`
- Check for port conflicts (default is 8000)
- Try `mkdocs serve -a 127.0.0.1:8001` to use a different port

### Deployment Not Updating

If changes don't appear on GitHub Pages:

1. Check that the workflow ran successfully in **Actions**
2. Verify GitHub Pages is enabled in repository settings
3. Wait a few minutes — GitHub Pages can take time to update
4. Clear your browser cache or try incognito mode
5. Check that `gh-pages` branch exists and has recent commits

## CI/CD Integration

The documentation deployment is integrated with your CI/CD pipeline:

- **Automatic** — Every push to `main` updates the docs
- **Manual** — Use `workflow_dispatch` to trigger deployment manually
- **Permissions** — The workflow has `contents: write` permission

### Trigger Manual Deployment

1. Go to **Actions** tab on GitHub
2. Select **Deploy Documentation** workflow
3. Click **Run workflow**
4. Select `main` branch
5. Click **Run workflow** button

## Best Practices

!!! tip "Documentation-Driven Development"
    Update documentation alongside code changes. Keep the docs in sync with your implementation.

!!! tip "Test Locally First"
    Always run `mkdocs serve` locally to preview changes before pushing.

!!! tip "Write Clear Examples"
    Include code examples, command outputs, and real-world use cases in your documentation.

!!! tip "Keep Navigation Simple"
    Don't nest too deeply — aim for 2-3 levels maximum in the navigation structure.
