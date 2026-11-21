// What Weightsmith is, how to use it, and every shortcut: one page, opened with "?".

import { Modal } from "./Modal";

const STEPS: [string, string][] = [
  ["Start", "Projects tab: start from a model, import a pretrained one, or create a blank project."],
  ["Build", "Select a node and click a library item: it is added after the selection, wired and sized from its input. Select a connection to insert into it instead."],
  ["Fix", "Problems show on the ports. Most have a fix button in the inspector."],
  ["Configure", "With nothing selected, the inspector's Data and Training tabs set the dataset and the optimiser; the checklist says what is still missing."],
  ["Stages", "The left rail follows the work: Model (1) builds the graph, Data (2) checks what the model will see, Train (3) sets the optimizer and runs, Runs (4) has the curves and comparisons, Diagnose (5) explains spikes."],
  ["Run", "Run (⌘↵) trains on the daemon from any stage; the strip at the bottom follows the run. Code next to Graph shows the generated model.py."],
  ["Explore", "Branch a variant to try an idea, compare runs with their graph diff, ablate a node, or export a run as plain PyTorch."],
];

const KEYS: [string, string][] = [
  ["⌘K", "add a node or run any command"],
  ["⌘↵", "run the current variant"],
  ["⌘Z / ⇧⌘Z", "undo / redo"],
  ["⌘C ⌘V ⌘D", "copy, paste, duplicate"],
  ["⌫", "delete (the chain stays connected)"],
  ["Enter", "edit the selected node"],
  ["⇧Enter", "open the selected block"],
  ["Esc", "deselect, then leave the block"],
  ["F", "fit the graph to the view"],
  ["M", "minimap on / off"],
  ["?", "this page"],
];

export function Help({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="How Weightsmith works" onClose={onClose} width={640} top={48}>
      <div className="grid gap-6 px-4 pb-4 sm:grid-cols-[1fr_220px]">
        <ol className="m-0 flex list-none flex-col gap-3 p-0">
          {STEPS.map(([h, t], i) => (
            <li key={h} className="grid grid-cols-[20px_1fr] gap-2">
              <span className="num text-label text-muted">{i + 1}</span>
              <span>
                <span className="font-medium">{h}. </span>
                <span className="text-muted">{t}</span>
              </span>
            </li>
          ))}
        </ol>
        <dl className="m-0 grid grid-cols-[auto_1fr] content-start gap-x-3 gap-y-1.5 text-small">
          {KEYS.map(([k, d]) => (
            <div key={k} className="contents">
              <dt>
                <kbd>{k}</kbd>
              </dt>
              <dd className="m-0 text-muted">{d}</dd>
            </div>
          ))}
        </dl>
      </div>
      <p className="m-0 border-t border-line px-4 py-2 text-label text-muted">
        The graph is the source of truth: the generated PyTorch (Code tab) always follows it, and every edit can be undone.
      </p>
    </Modal>
  );
}
