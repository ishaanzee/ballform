export type AnalysisMode = "form" | "one_on_one";
export type Handedness = "left" | "right";
export type RimBox = [number, number, number, number];
export type Metrics = Record<string, number | string | null>;

export interface GameAnalysis {
  score: number | null;
  confidence: number;
  status: string;
  metrics: Metrics;
  components: Record<string, number>;
  evidence: string[];
  limitations: string[];
  release_frame: number | null;
}

export interface ShotResult {
  number: number;
  start_s: number;
  release_s: number;
  end_s: number;
  outcome: string;
  outcome_confidence: number;
  evidence: string[];
  metrics: Metrics;
  cues: string[];
  game?: GameAnalysis | null;
}

export interface AnalysisResult {
  mode?: AnalysisMode;
  handedness?: Handedness;
  right_handed?: boolean;
  video: { duration_s: number; fps: number; resolution: string; analyzed_fps: number };
  camera_view: string;
  camera_view_confidence: number;
  shots: ShotResult[];
  diagnostics: { pose_frames: number; ball_detections: number; two_player_frames?: number; rim_marked: boolean };
  limitations: string[];
  annotated_video: string;
  game_summary?: {
    total_shots: number;
    scored_shots: number;
    mean_score: number | null;
    method: { version: string; label: string; formula: string; separation_component: string; contest_component: string; weights: Record<string, number>; note: string; [key: string]: unknown };
    limitations: string[];
  } | null;
}

export interface JobState {
  status: "queued" | "waiting" | "running" | "complete" | "failed";
  progress: number;
  message?: string;
  error?: string;
  result?: AnalysisResult;
}
