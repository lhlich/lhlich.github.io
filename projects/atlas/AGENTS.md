# Atlas publishing contract

Atlas is a self-contained project within the personal GitHub Pages website, located at `projects/atlas/`. Do not modify unrelated paths.

## Publish a document
1. Create a standalone HTML file or PDF in `projects/atlas/documents/` (nested directories are allowed).
2. Use relative links for CSS, scripts, images and navigation so the document works under GitHub Pages.
3. Add/update one entry in `projects/atlas/catalog.json` with: `id` (stable unique string), `title`, `description`, `topics` (array of strings), `format` (`html` or `pdf`), `href` (relative to Atlas root), and `updated` (YYYY-MM-DD).
4. Validate JSON and confirm every catalog URL points to a committed artifact. Test desktop and mobile readability.
5. For revisions, preserve stable document URLs and describe substantive changes in Git commit messages. Review AI-generated HTML/JS before publication.

## Publishing
Commit the document and updated catalog together to `main` (or through a reviewed PR). GitHub Pages already deploys this repository from its branch. No custom Actions workflow is needed.

The catalog is also the agent-facing discovery interface at `/projects/atlas/catalog.json`.

Do not publish private/confidential information. Avoid untrusted external scripts.
