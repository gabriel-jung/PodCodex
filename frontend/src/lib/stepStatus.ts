/** Shared provenance-shape check. The backend stamps `manual_edit: true`
 *  on a version whenever the user hand-edits segments in the SegmentEditor;
 *  UI components use this to flip colors from "needs review" to "reviewed". */
export function isManualEdit(provenance: unknown): boolean {
  return (provenance as { manual_edit?: unknown } | undefined)?.manual_edit === true;
}
