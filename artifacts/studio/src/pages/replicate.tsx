import { useState, useRef, useCallback, useEffect } from "react";
import { useLocation } from "wouter";
import {
  Upload, FileSpreadsheet, AlertTriangle, ChevronRight,
  RefreshCw, Play, X, File as FileIcon,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useGetJobStatus, useGenerateFromSpec } from "@workspace/api-client-react";

type Step = "upload" | "polling" | "review" | "generating";

const ACCEPT = ".csv,.xlsx,.xls,.json";
const ACCEPT_MIME = ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel", "application/json", ""];

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Replicate() {
  const [, navigate] = useLocation();
  const [step, setStep] = useState<Step>("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [spec, setSpec] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const generateMutation = useGenerateFromSpec();

  const { data: jobStatus } = useGetJobStatus(jobId!, {
    query: {
      enabled: step === "polling" && !!jobId,
      refetchInterval: step === "polling" && !!jobId ? 1500 : false,
    },
  });

  useEffect(() => {
    if (step !== "polling" || !jobStatus) return;
    if (jobStatus.status === "complete" && jobStatus.spec) {
      setSpec(jobStatus.spec);
      setStep("review");
    } else if (jobStatus.status === "failed") {
      setError(jobStatus.error || "Profiling failed");
      setStep("upload");
    }
  }, [jobStatus, step]);

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const arr = Array.from(incoming).filter(f => {
      const ext = "." + f.name.split(".").pop()!.toLowerCase();
      return ACCEPT.includes(ext);
    });
    if (!arr.length) {
      setError("No supported files found. Use CSV, XLSX, or JSON.");
      return;
    }
    setError(null);
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name));
      return [...prev, ...arr.filter(f => !names.has(f.name))];
    });
  }, []);

  const removeFile = (name: string) => {
    setFiles(prev => prev.filter(f => f.name !== name));
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      addFiles(e.target.files);
      e.target.value = "";
    }
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setError(null);
    setUploading(true);

    const fd = new FormData();
    for (const f of files) {
      fd.append("files", f, f.name);
    }

    try {
      const res = await fetch("/api/replicate", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(
          (data?.detail?.error) || (data?.detail) || data?.error || "Upload failed"
        );
      }
      setJobId(data.job_id);
      setStep("polling");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleGenerate = () => {
    if (!spec) return;
    setStep("generating");
    generateMutation.mutate(
      { data: { spec, format: "csv" } },
      {
        onSuccess: (data: any) => navigate(`/jobs/${data.job_id}`),
        onError: (err: any) => {
          setError(err?.data?.error || err?.message || "Generation failed");
          setStep("review");
        },
      }
    );
  };

  const reset = () => {
    setStep("upload");
    setFiles([]);
    setJobId(null);
    setSpec(null);
    setError(null);
  };

  const stepLabels = ["Upload", "Profile", "Review & Generate"];
  const stepKeys: Step[] = ["upload", "polling", "review"];
  const activeIdx = step === "generating" ? 2 : stepKeys.indexOf(step);

  return (
    <div className="container max-w-4xl py-8 space-y-6">
      <div className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight">Replicator</h1>
        <p className="text-muted-foreground">
          Upload one or more data files. The engine will profile each file's structure and generate a synthetic replica.
        </p>
      </div>

      <div className="flex items-center gap-2 text-sm">
        {stepLabels.map((label, i) => (
          <span key={label} className="flex items-center gap-2">
            {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
            <span className={`font-medium ${activeIdx === i ? "text-primary" : "text-muted-foreground"}`}>
              {i + 1}. {label}
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

      {/* ── Upload step ── */}
      {step === "upload" && (
        <div className="space-y-4">
          {/* Drop zone */}
          <div
            role="button"
            tabIndex={0}
            aria-label="Upload files"
            className={`rounded-xl border-2 border-dashed transition-colors cursor-pointer select-none
              ${isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/20"}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
          >
            <div className="flex flex-col items-center justify-center py-14 space-y-4 pointer-events-none">
              <div className={`p-4 rounded-full transition-colors ${isDragging ? "bg-primary/10" : "bg-muted"}`}>
                <Upload className={`h-8 w-8 ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <div className="text-center space-y-1">
                <p className="font-medium">
                  {isDragging ? "Drop files here" : "Drop files here, or click to browse"}
                </p>
                <p className="text-sm text-muted-foreground">
                  CSV · XLSX · JSON &nbsp;·&nbsp; Multiple files supported
                </p>
              </div>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept={ACCEPT}
            multiple
            onChange={handleInputChange}
          />

          {/* File list */}
          {files.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  {files.length} file{files.length !== 1 ? "s" : ""} selected
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 pt-0">
                {files.map(f => (
                  <div
                    key={f.name}
                    className="flex items-center gap-3 p-2 rounded-md border border-border/50 bg-muted/20"
                  >
                    <FileIcon className="h-4 w-4 text-primary shrink-0" />
                    <span className="font-mono text-sm flex-1 truncate">{f.name}</span>
                    <span className="text-xs text-muted-foreground shrink-0">{formatBytes(f.size)}</span>
                    <button
                      className="shrink-0 text-muted-foreground hover:text-destructive transition-colors"
                      onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                      aria-label={`Remove ${f.name}`}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                ))}
                <div className="flex justify-end pt-2">
                  <Button onClick={handleUpload} disabled={uploading} size="lg">
                    {uploading ? (
                      <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />Uploading…</>
                    ) : (
                      <><Upload className="h-4 w-4 mr-2" />Profile {files.length} file{files.length !== 1 ? "s" : ""}</>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* ── Polling step ── */}
      {step === "polling" && (
        <Card>
          <CardContent className="py-12 flex flex-col items-center gap-6">
            <RefreshCw className="h-10 w-10 text-primary animate-spin" />
            <div className="text-center space-y-1">
              <p className="font-semibold">Profiling {files.length} file{files.length !== 1 ? "s" : ""}…</p>
              <p className="text-sm text-muted-foreground">
                {jobStatus?.message || "Detecting types, distributions and constraints"}
              </p>
            </div>
            {jobStatus?.progress != null && (
              <div className="w-full max-w-sm">
                <Progress value={jobStatus.progress} className="h-1.5" />
                <p className="text-xs text-muted-foreground mt-1 text-right">
                  {Math.round(jobStatus.progress)}%
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Review / Generate step ── */}
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
              <CardDescription>Review the profiled spec before generating synthetic data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
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
                      <div
                        key={col.name}
                        className="flex items-center gap-1.5 text-xs p-1.5 rounded bg-muted/40 border border-border/50"
                      >
                        <span className="font-mono truncate">{col.name}</span>
                        <Badge className="text-[10px] px-1 py-0 shrink-0" variant="outline">{col.type}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <Button variant="outline" onClick={reset} disabled={step === "generating"}>
              Start over
            </Button>
            <Button size="lg" onClick={handleGenerate} disabled={step === "generating"}>
              {step === "generating" ? (
                <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />Generating…</>
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
