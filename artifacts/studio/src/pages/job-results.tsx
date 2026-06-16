import { useParams } from "wouter";
import { Link } from "wouter";
import { Download, RefreshCw, AlertTriangle, CheckCircle2, ArrowLeft, BarChart2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { RadialBarChart, RadialBar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from "recharts";
import { useGetJobStatus } from "@workspace/api-client-react";

export default function JobResults() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;

  const { data: job } = useGetJobStatus(jobId, {
    query: {
      enabled: !!jobId,
      refetchInterval: (query) => {
        const status = (query.state.data as any)?.status;
        return status === "pending" || status === "running" ? 1500 : false;
      },
    },
  });

  if (!job) {
    return (
      <div className="container max-w-4xl py-8 flex items-center justify-center min-h-[400px]">
        <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const isRunning = job.status === "pending" || job.status === "running";
  const result = job.result as any;
  const report = result?.fidelity_report;

  return (
    <div className="container max-w-5xl py-8 space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="space-y-0.5">
          <h1 className="text-2xl font-bold tracking-tight font-mono">{jobId.slice(0, 8)}</h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="capitalize">{job.mode} mode</span>
            <span>&bull;</span>
            <span>{new Date(job.created_at).toLocaleString()}</span>
          </div>
        </div>
        <div className="ml-auto">
          <Badge
            variant="outline"
            className={`capitalize ${
              job.status === "complete" ? "border-emerald-500/30 text-emerald-500" :
              job.status === "failed" ? "border-destructive/30 text-destructive" :
              "border-blue-500/30 text-blue-400"
            }`}
          >
            {job.status}
          </Badge>
        </div>
      </div>

      {isRunning && (
        <Card>
          <CardContent className="py-10 flex flex-col items-center gap-5">
            <RefreshCw className="h-8 w-8 text-primary animate-spin" />
            <div className="text-center space-y-1">
              <p className="font-semibold">{job.message || "Generating..."}</p>
              <p className="text-sm text-muted-foreground">This may take a few seconds</p>
            </div>
            {job.progress != null && (
              <div className="w-full max-w-sm">
                <Progress value={job.progress} className="h-1.5" />
                <p className="text-xs text-muted-foreground mt-1 text-right">{Math.round(job.progress)}%</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {job.status === "failed" && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{job.error || "Generation failed"}</AlertDescription>
        </Alert>
      )}

      {job.status === "complete" && result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="md:col-span-1">
              <CardContent className="pt-6 flex flex-col items-center justify-center gap-2">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Fidelity Score</p>
                <div className="w-32 h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                      cx="50%" cy="50%"
                      innerRadius="65%" outerRadius="90%"
                      data={[{ value: report?.overall_score ?? 0 }]}
                      startAngle={90} endAngle={90 - 360 * ((report?.overall_score ?? 0) / 100)}
                    >
                      <RadialBar dataKey="value" fill="hsl(var(--primary))" cornerRadius={4} />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-4xl font-bold font-mono">{report?.overall_score?.toFixed(1) ?? "—"}</p>
                <p className="text-xs text-muted-foreground">out of 100</p>
              </CardContent>
            </Card>
            <div className="md:col-span-2 grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="pt-5">
                  <p className="text-xs font-medium text-muted-foreground">Tables</p>
                  <p className="text-3xl font-bold mt-1">{result.tables?.length ?? 0}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-5">
                  <p className="text-xs font-medium text-muted-foreground">Total Rows</p>
                  <p className="text-3xl font-bold mt-1">{result.row_count_total?.toLocaleString() ?? 0}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-5">
                  <p className="text-xs font-medium text-muted-foreground">Referential Integrity</p>
                  <p className="text-3xl font-bold mt-1">{report?.referential_integrity?.toFixed(1) ?? "100.0"}%</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-5">
                  <p className="text-xs font-medium text-muted-foreground">Constraint Pass Rate</p>
                  <p className="text-3xl font-bold mt-1">{report?.constraint_pass_rate?.toFixed(1) ?? "100.0"}%</p>
                </CardContent>
              </Card>
            </div>
          </div>

          {report?.per_table?.map((tableReport: any) => (
            <FidelityTableSection key={tableReport.table_name} tableReport={tableReport} />
          ))}

          {result.tables?.map((t: any) => (
            <TablePreview key={t.name} table={t} />
          ))}

          <div className="flex items-center justify-end gap-3 pt-2">
            <a href={`/api/download/${jobId}/csv`} download>
              <Button variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Download CSV
              </Button>
            </a>
            <a href={`/api/download/${jobId}/xlsx`} download>
              <Button>
                <Download className="h-4 w-4 mr-2" />
                Download XLSX
              </Button>
            </a>
          </div>
        </>
      )}
    </div>
  );
}

function FidelityTableSection({ tableReport }: { tableReport: any }) {
  const chartData = tableReport.per_column
    .filter((c: any) => c.ks_statistic != null || c.category_overlap != null)
    .slice(0, 12)
    .map((c: any) => ({
      name: c.column_name,
      score: c.ks_statistic != null
        ? Math.round((1 - c.ks_statistic) * 100)
        : Math.round((c.category_overlap ?? 0) * 100),
    }));

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <BarChart2 className="h-4 w-4 text-primary" />
          <span className="font-mono">{tableReport.table_name}</span>
          <Badge variant="outline" className="text-xs ml-2">
            Constraints: {tableReport.constraint_pass_rate.toFixed(0)}%
          </Badge>
        </CardTitle>
        <CardDescription>Per-column fidelity (higher = closer to source distribution)</CardDescription>
      </CardHeader>
      {chartData.length > 0 && (
        <CardContent>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={chartData} margin={{ top: 0, right: 0, bottom: 20, left: 0 }}>
              <XAxis dataKey="name" tick={{ fontSize: 10, fontFamily: "monospace" }} angle={-30} textAnchor="end" interval={0} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={28} />
              <Tooltip
                formatter={(v: any) => [`${v}%`, "Score"]}
                contentStyle={{ fontSize: 12, background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 6 }}
              />
              <Bar dataKey="score" radius={[3, 3, 0, 0]}>
                {chartData.map((_: any, i: number) => (
                  <Cell key={i} fill={`hsl(var(--primary))`} fillOpacity={0.7 + (i % 3) * 0.1} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      )}
    </Card>
  );
}

function TablePreview({ table }: { table: any }) {
  if (!table.preview?.length) return null;
  const cols = table.columns.slice(0, 6);
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-mono flex items-center gap-2">
          {table.name}
          <Badge variant="secondary" className="text-xs">{table.row_count.toLocaleString()} rows</Badge>
        </CardTitle>
        <CardDescription>First 10 rows preview</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                {cols.map((c: string) => (
                  <TableHead key={c} className="font-mono text-xs">{c}</TableHead>
                ))}
                {table.columns.length > 6 && <TableHead className="text-muted-foreground text-xs">+{table.columns.length - 6} more</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody>
              {table.preview.slice(0, 10).map((row: any, i: number) => (
                <TableRow key={i}>
                  {cols.map((c: string) => (
                    <TableCell key={c} className="font-mono text-xs max-w-[180px] truncate">
                      {row[c] == null ? <span className="text-muted-foreground italic">null</span> : String(row[c])}
                    </TableCell>
                  ))}
                  {table.columns.length > 6 && <TableCell />}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
