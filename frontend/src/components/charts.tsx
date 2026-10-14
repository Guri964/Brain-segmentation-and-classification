import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ReactElement } from "react";

type Row = Record<string, string | number | null | undefined>;
const AXIS = { stroke: "currentColor", fontSize: 11, opacity: 0.65 };
const GRID = "rgba(120, 130, 150, 0.25)";
const PALETTE = [
  "#0ea5e9",
  "#14b8a6",
  "#f59e0b",
  "#8b5cf6",
  "#ef4444",
  "#22c55e",
  "#6366f1",
  "#ec4899",
];
export function ChartFrame({
  children,
  height = 260,
}: {
  children: ReactElement;
  height?: number;
}) {
  return (
    <div
      className="text-clinical-700 dark:text-clinical-200"
      style={{ width: "100%", height }}
    >
      <ResponsiveContainer>{children}</ResponsiveContainer>
    </div>
  );
}
export function FeatureBarChart({
  data,
  dataKey = "value",
  color = "#0ea5e9",
  height = 260,
}: {
  data: Row[];
  dataKey?: string;
  color?: string;
  height?: number;
}) {
  return (
    <ChartFrame height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 48, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="label"
          tick={AXIS}
          angle={-40}
          textAnchor="end"
          interval={0}
          height={60}
        />
        <YAxis tick={AXIS} width={56} />
        <RechartsTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Bar dataKey={dataKey} fill={color} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}
export function MetricComparisonChart({
  data,
  keys,
  height = 300,
}: {
  data: Row[];
  keys: string[];
  height?: number;
}) {
  return (
    <ChartFrame height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 48, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="label"
          tick={AXIS}
          angle={-30}
          textAnchor="end"
          interval={0}
          height={60}
        />
        <YAxis tick={AXIS} width={56} domain={[0, 1]} />
        <RechartsTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {keys.map((key, index) => (
          <Bar
            dataKey={key}
            fill={PALETTE[index % PALETTE.length]}
            radius={[3, 3, 0, 0]}
            key={key}
          />
        ))}
      </BarChart>
    </ChartFrame>
  );
}
export function TrainingCurve({
  points,
  xKey = "epoch",
  lines,
}: {
  points: Row[];
  xKey?: string;
  lines: string[];
}) {
  return (
    <ChartFrame height={280}>
      <LineChart
        data={points}
        margin={{ top: 8, right: 16, bottom: 24, left: 4 }}
      >
        <CartesianGrid stroke={GRID} />
        <XAxis
          dataKey={xKey}
          tick={AXIS}
          label={{ value: xKey, position: "insideBottomRight", fontSize: 11 }}
        />
        <YAxis tick={AXIS} width={56} />
        <RechartsTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {lines.map((line, index) => (
          <Line
            type="monotone"
            dataKey={line}
            stroke={PALETTE[index % PALETTE.length]}
            dot={false}
            strokeWidth={2}
            key={line}
          />
        ))}
      </LineChart>
    </ChartFrame>
  );
}
export function FeatureScatter({
  data,
  xKey,
  yKey,
}: {
  data: Row[];
  xKey: string;
  yKey: string;
}) {
  return (
    <ChartFrame height={280}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis dataKey={xKey} name={xKey} tick={AXIS} type="number" />
        <YAxis
          dataKey={yKey}
          name={yKey}
          tick={AXIS}
          width={56}
          type="number"
        />
        <ZAxis range={[40, 40]} />
        <RechartsTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Scatter data={data} fill="#14b8a6" />
      </ScatterChart>
    </ChartFrame>
  );
}
export function CategoryRadar({
  data,
}: {
  data: Array<{ category: string; value: number }>;
}) {
  return (
    <ChartFrame height={280}>
      <RadarChart data={data} outerRadius="70%">
        <PolarGrid stroke={GRID} />
        <PolarAngleAxis dataKey="category" tick={{ fontSize: 11 }} />
        <PolarRadiusAxis tick={{ fontSize: 10 }} />
        <Radar
          dataKey="value"
          stroke="#8b5cf6"
          fill="#8b5cf6"
          fillOpacity={0.35}
        />
        <RechartsTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
      </RadarChart>
    </ChartFrame>
  );
}
export { Cell, PALETTE };
