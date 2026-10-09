# My Doc Hub publishing contract

My Doc Hub is a self-contained project within the personal GitHub Pages website, located at `projects/atlas/`. Do not modify unrelated paths.

## Publish a document
1. Create a standalone HTML file or PDF in `projects/atlas/documents/` (nested directories are allowed).
2. Use relative links for CSS, scripts, images and navigation so the document works under GitHub Pages.
3. Add/update one entry in `projects/atlas/catalog.json` with: `id` (stable unique string), `title`, `description`, `topics` (array of strings), `format` (`html` or `pdf`), `href` (relative to the hub root), and `updated` (YYYY-MM-DD).
4. Validate JSON and confirm every catalog URL points to a committed artifact. Test desktop and mobile readability.
5. For revisions, preserve stable document URLs and describe substantive changes in Git commit messages. Review AI-generated HTML/JS before publication.

## Publishing
Optional catalog fields: `kind` (e.g. Study guide), `details` (short factual summary of scope), and `contents` (array of subject labels). These enrich library entries without requiring changes to the documents. Omit fields that do not apply; do not invent section counts or review status.

Commit the document and updated catalog together to `main` (or through a reviewed PR). GitHub Pages already deploys this repository from its branch. No custom Actions workflow is needed.

The catalog is also the agent-facing discovery interface at `/projects/atlas/catalog.json`.

Do not publish private/confidential information. Avoid untrusted external scripts.

## Shared reader navigation
The hub opens catalog entries through `reader.html?id=<stable-id>`. The reader provides the return-to-hub and Open original links for HTML and PDF documents. Do not add hub navigation or shared navigation scripts to individual documents. Keep catalog `href` values pointing to original artifacts. Direct artifact URLs remain standalone; use the reader URL when sharing a document with hub navigation. PDFs also have an Open original fallback for browsers without embedded PDF support.

Documents are trusted, reviewed site content, not sandboxed third-party uploads. Preserve each document's independent styling, scripts, and relative assets.
