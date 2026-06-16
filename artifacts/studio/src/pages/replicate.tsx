import { useState, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { Upload, FileSpreadsheet, AlertTriangle, ChevronRight, RefreshCw, Play } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useGetJobStatus, useGenerateFromSpec } from "@workspace/api-client-react";

type Step = "upload" | "polling" | "review" | "generating";

export default function Replicate() {
  const [, navigate] = useLocation();
  const [step, setStep] = useState<Step>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
  const [spec, setSpec] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const generateMutation = useGenerateFromSpec();

  const shouldPoll = step === "polling" && !!jobId;
  const { data: jobStatus } = useGetJobStatus(jobId!, {
    query: {
      enabled: shouldPoll,
      refetchInterval: shouldPoll ? 1500 : false,
    },
  });

  if (shouldPoll && jobStatus) {
    if (jobStatus.status === "complete" && jobStatus.spec) {
      setSpec(jobStatus.spec);
      setStep("review");
    } else if (jobStatus.status === "failed") {
      setError(jobStatus.error || "Job failed");
      setStep("upload");
    }
  }

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/replicate", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setJobId(data.job_id);
      setStep("polling");
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleGenerate = () => {
    if (!spec) return;
    setStep("generating");
    generateMutation.mutate(
      { data: { spec, format: "csv" } },
      {
        onSuccess: (data: any) => {
          navigate(`/jobs/${data.job_id}`);
        },
        onError: (err: any) => {
          setError(err?.data?.error || "Generation failed");
          setStep("review");
        },
      }
    );
  };

  return (
    <div className="container max-w-4xl py-8 space-y-6">
      <div className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight">Replicator</h1>
        <p className="text-muted-foreground">
          Upload a data file to profile its structure and generate a synthetic replica.
        </p>
      </div>

      <div className="flex items-center gap-2 text-sm">
        {(["upload", "polling", "review"] as Step[]).map((s, i) => (
          <span key={s} className="flex items-center gap-2">
            {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
            <span className={`font-medium ${step === s || (step === "generating" && s === "review") ? "text-primary" : "text-muted-foreground"}`}>
              {i + 1}. {s === "upload" ? "Upload" : s === "polling" ? "Profile" : "Review & Generate"}
            </span>
          </span>
        ))}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {step === "upload" && (
        <Card
          className={`border-2 border-dashed transition-colors cursor-pointer ${isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <CardContent className="flex flex-col items-center justify-center py-16 space-y-4">
            <div className="p-4 rounded-full bg-muted">
              <Upload className="h-8 w-8 text-muted-foreground" />
            </div>
            <div className="text-center space-y-1">
              <p className="font-medium">Drop your file here, or click to browse</p>
              <p className="text-sm text-muted-foreground">Supports CSV, XLSX, JSON</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".csv,.xlsx,.xls,.json"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </CardContent>
        </Card>
      )}

      {step === "polling" && (
        <Card>
          <CardContent className="py-12 flex flex-col items-center gap-6">
            <RefreshCw className="h-10 w-10 text-primary animate-spin" />
            <div className="text-center space-y-1">
              <p className="font-semibold">Profiling file...</p>
              <p className="text-sm text-muted-foreground">{jobStatus?.message || "Detecting types, distributions and constraints"}</p>
            </div>
            {jobStatus?.progress != null && (
              <div className="w-full max-w-sm">
                <Progress value={jobStatus.progress} className="h-1.5" />
                <p className="text-xs text-muted-foreground mt-1 text-right">{Math.round(jobStatus.progress)}%</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {(step === "review" || step === "generating") && spec && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Inferred Schema</CardTitle>
                <div className="flex gap-2">
                  <Badge variant="outline">{spec.tables?.length ?? 0} table(s)</Badge>
                  <Badge variant="outline">{spec.relationships?.length ?? 0} relationship(s)</Badge>
                </div>
              </div>
              <CardDescription>Review the inferred spec before generating</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {spec.tables?.map((table: any) => (
                <div key={table.name} className="space-y-2">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="h-4 w-4 text-primary" />
                    <span className="font-mono font-semibold">{table.name}</span>
                    <Badge variant="secondary" className="text-xs">{table.row_count} rows</Badge>
                    {table.primary_key && (
                      <Badge variant="outline" className="text-xs font-mono">PK: {table.primary_key}</Badge>
                    )}
                  </div>
                  <div className="ml-6 grid grid-cols-2 md:grid-cols-3 gap-1.5">
                    {table.columns?.map((col: any) => (
                      <div key={col.name} className="flex items-center gap-1.5 text-xs p-1.5 rounded bg-muted/40 border border-border/50">
                        <span className="font-mono truncate">{col.name}</span>
                        <Badge className="text-[10px] px-1 py-0 shrink-0" variant="outline">{col.type}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <Button
              size="lg"
              onClick={handleGenerate}
              disabled={step === "generating"}
            >
              {step === "generating" ? (
                <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />Generating...</>
              ) : (
                <><Play className="h-4 w-4 mr-2" />Generate Synthetic Data</>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
