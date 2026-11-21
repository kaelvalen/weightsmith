// Shapes served by trainscope's run-keyed API (trainscope/trainscope/ui/server.py).

export interface GlobalRow {
  step: number;
  loss?: number | null;
  grad_norm_before_clip?: number | null;
  grad_nan_inf_ratio?: number | null;
  spike_score?: number | null;
  is_spike?: boolean;
  lr?: number | null;
  [k: string]: unknown;
}

export interface SpikeEvent {
  id: string;
  startStep: number;
  endStep: number;
  count: number;
  steps: number[];
  peakStep: number;
  peakLoss: number | null;
  earlyWarningWindow: number;
  label: string;
}

export interface LayerMatrix {
  layers: string[];
  z: (number | null)[][];
  values: (number | null)[][];
  bucket_starts?: number[];
  [k: string]: unknown;
}

export interface RunSummary {
  name: string;
  model_name?: string | null;
  last_step?: number | null;
  n_global_rows?: number | null;
  spike_count: number;
  last_loss?: number | null;
  last_loss_finite?: boolean | null;
  [k: string]: unknown;
}
