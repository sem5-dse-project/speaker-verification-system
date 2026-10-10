import React from 'react';
import { Composition } from 'remotion';
import { MainVideo } from './Video';
import { TOTAL_FRAMES, FPS, SCENE_STARTS, SCENE_DURATIONS, CLIP_DURATIONS, clipFrames } from './constants';

// ── Scene components ──────────────────────────────────────────────────────────
import { Intro } from './scenes/Intro';
import { Login } from './scenes/Login';
import { AudioInput } from './scenes/AudioInput';
import { VAD } from './scenes/VAD';
import { ReplayDetection } from './scenes/ReplayDetection';
import { ModelPipeline } from './scenes/ModelPipeline';
import { EcapaIntro } from './scenes/EcapaIntro';
import { SpeechFrames } from './scenes/SpeechFrames';
import { EcapaNetwork } from './scenes/EcapaNetwork';
import { FeatureAttention } from './scenes/FeatureAttention';
import { MultiScaleFeatures } from './scenes/MultiScaleFeatures';
import { AttentivePooling } from './scenes/AttentivePooling';
import { SpeakerEmbedding } from './scenes/SpeakerEmbedding';
import { EmbeddingComparison } from './scenes/EmbeddingComparison';
import { AuthResult } from './scenes/AuthResult';
import { Architecture } from './scenes/Architecture';
import { End } from './scenes/End';

// ── Chapter markers for the combined video ────────────────────────────────────
const COMBINED_CHAPTERS = [
  { title: '01 · Intro',                          key: 'intro'              },
  { title: '02 · Login Demo',                     key: 'login'              },
  { title: '03 · Audio Input',                    key: 'audioInput'         },
  { title: '04 · Silero VAD',                     key: 'vad'                },
  { title: '05 · Replay Detection',               key: 'replay'             },
  { title: '06 · Full ML Pipeline',               key: 'modelPipeline'      },
  { title: '07 · ECAPA-TDNN Intro',               key: 'ecapaIntro'         },
  { title: '08 · Speech Frames',                  key: 'speechFrames'       },
  { title: '09 · ECAPA Network',                  key: 'ecapaNetwork'       },
  { title: '10 · Feature Attention (SE)',          key: 'featureAttention'   },
  { title: '11 · Multi-Scale Features (Res2Net)', key: 'multiScale'         },
  { title: '12 · Attentive Pooling',              key: 'attentivePooling'   },
  { title: '13 · Speaker Embedding (192-dim)',    key: 'speakerEmbedding'   },
  { title: '14 · Embedding Comparison',           key: 'embeddingComparison'},
  { title: '15 · Auth Result',                    key: 'authResult'         },
  { title: '16 · System Architecture',            key: 'architecture'       },
  { title: '17 · End',                            key: 'end'                },
] as const;

const combinedChapters = COMBINED_CHAPTERS.map((c) => ({
  title: c.title,
  startInFrames: SCENE_STARTS[c.key as keyof typeof SCENE_DURATIONS],
}));

// ── Individual clip definitions ───────────────────────────────────────────────
// Each entry maps to one standalone <Composition> rendered as its own MP4.
const CLIP_DEFS = [
  { id: 'clip01-Intro',               component: Intro              },
  { id: 'clip02-Login',               component: Login              },
  { id: 'clip03-AudioInput',          component: AudioInput         },
  { id: 'clip04-VAD',                 component: VAD                },
  { id: 'clip05-Replay',              component: ReplayDetection    },
  { id: 'clip06-ModelPipeline',       component: ModelPipeline      },
  { id: 'clip07-EcapaIntro',          component: EcapaIntro         },
  { id: 'clip08-SpeechFrames',        component: SpeechFrames       },
  { id: 'clip09-EcapaNetwork',        component: EcapaNetwork       },
  { id: 'clip10-FeatureAttention',    component: FeatureAttention   },
  { id: 'clip11-MultiScale',          component: MultiScaleFeatures },
  { id: 'clip12-AttentivePooling',    component: AttentivePooling   },
  { id: 'clip13-SpeakerEmbedding',    component: SpeakerEmbedding   },
  { id: 'clip14-EmbeddingComparison', component: EmbeddingComparison},
  { id: 'clip15-AuthResult',          component: AuthResult         },
  { id: 'clip16-Architecture',        component: Architecture       },
  { id: 'clip17-End',                 component: End                },
] as const;

// ── Root ─────────────────────────────────────────────────────────────────────
export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* ── Combined 95-second video ── */}
      <Composition
        id="VoiceAuthDemo"
        component={MainVideo}
        durationInFrames={TOTAL_FRAMES}
        fps={FPS}
        width={1920}
        height={1080}
        calculateMetadata={() => ({
          props: {},
          durationInFrames: TOTAL_FRAMES,
          fps: FPS,
          width: 1920,
          height: 1080,
          chapters: combinedChapters,
        })}
      />

      {/* ── Individual scene clips ── */}
      {CLIP_DEFS.map(({ id, component: Comp }) => {
        const seconds = CLIP_DURATIONS[id] ?? 20;
        const frames  = clipFrames(seconds);
        return (
          <Composition
            key={id}
            id={id}
            component={Comp as React.ComponentType<Record<string, unknown>>}
            durationInFrames={frames}
            fps={FPS}
            width={1920}
            height={1080}
          />
        );
      })}
    </>
  );
};
