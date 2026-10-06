import React from "react";
import { Composition } from "remotion";
import { FPS, totalDuration } from "./timeline";
import { VoiceAuthDemo } from "./Video";

export const RemotionRoot: React.FC = () => (
  <Composition
    id="VoiceAuthDemo"
    component={VoiceAuthDemo}
    durationInFrames={totalDuration}
    fps={FPS}
    width={1920}
    height={1080}
  />
);
